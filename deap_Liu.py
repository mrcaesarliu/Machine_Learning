# Genetic Programming code 
# Writ by LiuBW
# Time:2021/3/25
# version 1.0.0
'''
遗传规划发算法包，计算的输入数据仅支持 int、float 格式
计算表达式并、Fitness 计算不包含在此包当中

'''

import copy
import math
import random
import re
import sys
import warnings
import numpy as np

from collections import defaultdict, deque
from functools import partial, wraps
from inspect import isclass
from operator import eq, lt,attrgetter
#===========================================
# 初始化函数集合、终端集合
#===========================================

class Primitive:
    '''
    创建算子节点
    '''
    def __init__(self, name, args):
        self.name = name
        self.arity = len(args)
        self.args = args
        args = ", ".join(map("{{{0}}}".format, range(self.arity)))
        self.seq = "{name}({args})".format(name=self.name, args=args)
        pass

    def format(self, *args):
        
        return self.seq.format(*args)
    pass
class Terminal:
    '''
    创建叶子节点
    '''
    def __init__(self, terminal):
        self.name = terminal
        self.arity = 0

    def format(self):
        return self.name


class PrimitiveSet:
    '''
    创建树的集合
    '''
    def __init__(self):
        self.primitives = []
        self.terminals = []
        pass
    
    def AddPrim(self, primitive, args):
        prim = Primitive(primitive, args)
        self.primitives.append(prim)
        pass
    
    def AddTerm(self, terminal):
        term = Terminal(terminal)
        self.terminals.append(term)
        pass
    
    pass


#=====================================================================
# 生成随机二叉树
#=====================================================================

def condition(height, depth):
    """Expression generation stops when the depth is equal to height."""
    return depth == height

def generate(pset, min_, max_):
    '''
    生成树算法
    '''
    expr = []
    height = random.randint(min_, max_)
    stack = [(0)]
    
    while len(stack) != 0:
        
        depth = stack.pop()
        if condition(height, depth):
            term = random.choice(pset.terminals)
            if isclass(term):
                term = term()
                pass
            expr.append(term)
            pass
        else:
            prim = random.choice(pset.primitives)
            expr.append(prim)
            for arg in range(prim.arity):
                stack.append((depth + 1))
                pass
            pass
        
    return expr

#========================================================
#二叉树 list表达式 变成树状结构(可以搜索、遍历的二叉树结构)
#========================================================

class Tree(list):
    '''
    Expr List 初始化变成 Tree
    '''
    
    def __init__(self, expr):
        list.__init__(self, expr)
        self.pre_fitness_c = np.nan
        self.pre_fitness_m = np.nan
        self.mutation_info = (np.nan, np.nan)
        self.crossover_info = (np.nan, np.nan)
        
        
        pass
    
    
    def __str__(self):
        '''
        把 expr 变成可阅读的表达式
        '''
        string = ""
        stack = []
        for node in self:

            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args_ = stack.pop()
                string = prim.format(*args_)
                if len(stack) == 0:
                    break  
                stack[-1][1].append(string)
                pass
            pass

        return string
    
    def search_node(self, begin):
        '''
        搜寻节点下面的部分
        '''
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)
    
    @property
    def height(self):
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth    
    
    pass


#=====================================
# crossover, mutation
#=====================================
class Crossover:
    
    def __init__(self, PrimSet):
        self.PrimSet = PrimSet
        pass


    def crossover_process(self, ind_1, ind_2, slice1, slice2):
        '''
        对于替换掉的部分进行中间过程的计算
        
        注意核对结果的正确性
        '''
        
        ind_1_org = copy.deepcopy(ind_1)
        ind_2_org = copy.deepcopy(ind_2)
        # 用随机的leaves 替换被截掉的部分
        
        ind_1_part = copy.deepcopy(ind_1_org[slice1])
        ind_1_part = Tree(ind_1_part)
        
        
        ind_2_part = copy.deepcopy(ind_2_org[slice2])
        ind_2_part = Tree(ind_2_part)
        #pdb.set_trace()
        
        ind_1_org[slice1] = [random.choice(self.PrimSet.terminals)]
        ind_2_org[slice2] = [random.choice(self.PrimSet.terminals)]


        return ind_1_org, ind_2_org, ind_1_part, ind_2_part


    def crossover_1(self, ind_1, ind_2):
        '''
        单点crossover
        '''
        len_ind_1 = len(ind_1)
        len_ind_2 = len(ind_2)

        if len_ind_1<2 or len_ind_2<2:
            return ind_1, ind_2
        node_list_1 = range(1, len_ind_1)
        node_list_2 = range(1, len_ind_2)

        index_1 = random.choice(node_list_1)
        index_2 = random.choice(node_list_2)

        slice1 = ind_1.search_node(index_1)
        slice2 = ind_2.search_node(index_2)
        #  对于改变的list 进行计算、分解

        ind_1_org, ind_2_org, ind_1_part, ind_2_part = self.crossover_process(ind_1, ind_2, slice1, slice2)
        ind_1[slice1], ind_2[slice2] = ind_2[slice2], ind_1[slice1]
        return ind_1, ind_2, ind_1_org, ind_2_org, ind_1_part, ind_2_part
    
    
    
    def crossover_2(self, ind_1, ind_2, termpb):
        '''
        非leaves crossover
        '''
        
        if len(ind_1) < 2 or len(ind_2) < 2:
            return ind_1, ind_2

        # 决定crossover 的节点是 Primitive 还是 Terminal
        terminal_op = partial(eq, 0)
        primitive_op = partial(lt, 0)
        arity_op1 = terminal_op if random.random() < termpb else primitive_op
        arity_op2 = terminal_op if random.random() < termpb else primitive_op

        # 把符合条件的节点变成list
        types1 = []
        types2 = []

        for idx, node in enumerate(ind_1[1:], 1):
            if arity_op1(node.arity):
                types1.append(idx)

        for idx, node in enumerate(ind_2[1:], 1):
            if arity_op2(node.arity):
                types2.append(idx)



        # pdb.set_trace()

        # 随机选取节点
        index1 = random.choice(types1)
        index2 = random.choice(types2)

        slice1 = ind_1.search_node(index1)
        slice2 = ind_2.search_node(index2)
        
        ind_1_org, ind_2_org, ind_1_part, ind_2_part = self.crossover_process(ind_1, ind_2, slice1, slice2)
        
        ind_1[slice1], ind_2[slice2] = ind_2[slice2], ind_1[slice1]

        return ind_1, ind_2, ind_1_org, ind_2_org, ind_1_part, ind_2_part
    
    
    pass


class Mutation:
    '''
    对于表达式树进行变异
    '''
    def __init__(self):
        
        pass

    def mutUniform(self, individual,mutation_param, PrimSet):
        '''
        基础方式变异：
        在树状结构里面随机选取
        '''
        min_,max_ = mutation_param
        index = random.randrange(len(individual))
        slice_ = individual.search_node(index)
        replace_expr = generate(PrimSet, min_,max_)
        divd_expr = copy.deepcopy(individual[slice_])
        individual[slice_] = replace_expr
        replace_expr_ = copy.deepcopy(replace_expr)
        replace_tree = Tree(replace_expr_)
        divd_tree = Tree(divd_expr) 
        
        return individual, replace_tree, divd_tree
    pass

#=====================================
# selection 
#=====================================
def selRandom(individuals, k):
    '''
    在population 里面随机选取K个individual
    '''
    return [random.choice(individuals) for i in range(k)]


def selBest(individuals, k, fit_attr="fitness"):
    '''
    挑选最好的K个
    '''
    return sorted(individuals, key=attrgetter(fit_attr), reverse=True)[:k]


def selWorst(individuals, k, fit_attr="fitness"):
    '''
    挑选最差的K个个体
    '''
    return sorted(individuals, key=attrgetter(fit_attr))[:k]

def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    '''
    运行K此，每次从population 里面随机选取 tournsize 个，只取最大的样本那个
    '''
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


#=========================================
# Clone, Infonation Process
#=========================================

def Clone(pop):
    '''
    对输入的种群进行复制
    '''
    
    pop_clone = [copy.deepcopy(ind) for ind in pop]
    
    return pop_clone


#=========================================
# 对算法部分进行封装
#=========================================
class algorithm:
    
    def __init__(self):
        self.PrimSet = PrimitiveSet()
        
        pass
    
    def Init_Prim(self, FuncSet, leaveSet):
        '''
        为初始的PrimitiveSet 添加函数和叶子
        '''
        for func in FuncSet:
            self.PrimSet.AddPrim(func[0],func[1])
            pass
        
        for leave in leaveSet:
            self.PrimSet.AddTerm(leave)
            pass
        
        pass
    def Create_population(self, pop_num, min_depth, max_depth):
        
        pop = []
        for i in range(pop_num):
            expr = generate(self.PrimSet, min_depth, max_depth)
            Tree_ = Tree(expr)
            pop.append(Tree_)
            pass
        return pop
    
    def select(self, pop, fitness, select_num):
        '''
        对population 进行筛选:
        select_type: 1,2
        1. select best
        2. select 
        '''
        
        for ind, fit in zip(pop, fitness):
            ind.fitness = fit
            pass
        Select_1 = selBest(pop, select_num)
        
        return Select_1
    
    def Crossover(self, population):
        '''
        对于输入的种群进行crossover
        return: crossover后的pop,crossover之前主干的expr,crossover拼接上来的expr
        '''
        pop = copy.deepcopy(population)
        random.shuffle(pop)
        
        cross_over = Crossover(self.PrimSet)
        org_ = []
        part_ = [] 
        # 保留上一代的fitness
        for ind_ in pop:
            ind_.pre_fitness_c = ind_.fitness
            pass
        
        for i in range(1, len(pop),2):
            ind1 = pop[i-1]
            ind2 = pop[i]
            ind_1, ind_2, ind_1_org, ind_2_org, ind_1_part, ind_2_part = cross_over.crossover_2(ind1, ind2,0)
            try:
                ind_1.genetic = (ind1.genetic + ind2.genetic)/2
                ind_2.genetic = (ind1.genetic + ind2.genetic)/2
            except:
                print('上一代的genetic 没有标记')
                pdb.set_trace()
                pass
            org_.append(ind_1_org)
            org_.append(ind_2_org)
            part_.append(ind_2_part)
            part_.append(ind_1_part)
            pass
        return pop, org_, part_
    
    def population_info(self, pop, fitness_list):
        '''
        对种群的info进行赋值
        '''
        if len(pop)!=len(fitness_list):
            # fitness 包含了其他信息
            if len(pop)==len(fitness_list[0]):
                # 给 fitness 进行赋值
                for ind,fit in zip(pop, fitness_list[0]):
                    ind.fitness = fit
                    pass
                for j in range(1,len(fitness_list)):
                    if j==1:
                        info_ls = np.array(fitness_list[j]).reshape(-1)
                    else:
                        s_info =  np.array(fitness_list[j]).reshape(-1)
                        info_ls = np.vstack([info_ls, s_info])
                        pass
                    pass
                info_ls_ = info_ls.T
                for k in range(len(pop)):
                    pop[k].other_info = tuple(info_ls_[k,:]) 
                    pass
                
            else:
                print('fitness 与pop 数量对不上')
                pass
        else:
            for ind,fit in zip(pop, fitness_list):
                ind.fitness = fit
                pass 
            pass
        return pop
    
    def genetic_label(self, pop, fitness):
        '''
        给种群的基因进行标记
        '''
        up_thresh= np.percentile(fitness,67)
        down_thresh = np.percentile(fitness,33)
        
        for ind,fit in zip(pop, fitness):
            if fit>=up_thresh:
                ind.genetic =1 
            elif fit<=down_thresh:
                ind.genetic =-1
            else:
                ind.genetic =0
            pass
        
        return pop
    
    
    def cross_info(self, pop, pop_fitness ,org_fitness, part_fitness):
        '''
        对于 crossover
        '''
        
        for ind, ind_fit, org_fit, part_fit in zip(pop, pop_fitness ,org_fitness, part_fitness):
            ind.fitness = ind_fit
            ind.crossover_info = (org_fit, part_fit)
            pass
        
        return pop
    
    def Mutation(self, population, mutation_param):
        '''
        对于输入的种群进行Mutation
        individual, replace_tree, divd_expr
        '''
        pop = copy.deepcopy(population)

        
        Mutation_ = Mutation()
        Replace_ = []
        Split = []
        for ind_m in pop:
            ind_m.pre_fitness_m =  ind_m.fitness
            pass        
        
        for i in range(len(pop)):
            individual = pop[i]
            ind_, replace_tree, divd_expr = Mutation_.mutUniform(individual,mutation_param, self.PrimSet)
            Replace_.append(replace_tree)
            Split.append(divd_expr)
            pass
        
        return pop, Replace_, Split
    
    def mutation_info(self, pop, pop_fitness ,Replace_fitness, Split_fitness):
        '''
        对于
        '''
        for ind, ind_fit, replace_fit, split_fit in zip(pop, pop_fitness ,Replace_fitness, Split_fitness):
            ind.fitness = ind_fit
            ind.mutation_info = (replace_fit, split_fit)
            pass
        
        return pop
    pass




