import pandas as pd
from statsmodels import regression
import statsmodels.api as sm
import pyodbc
import random
import numpy as np
import warnings
import itertools
import statistics
import operator
import math
import random
import numpy
import numpy as np
import sys
try:
    import pickle as pickle
except ImportError:
    import pickle

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from scipy.stats.stats import pearsonr


### Collect all the outstanding alpha factors

class HallOfFame(object):
    """The hall of fame contains the best individual that ever lived in the
    population during the evolution. It is lexicographically sorted at all
    time so that the first element of the hall of fame is the individual that
    has the best first fitness value ever seen, according to the weights
    provided to the fitness at creation time.

    The insertion is made so that old individuals have priority on new
    individuals. A single copy of each individual is kept at all time, the
    equivalence between two individuals is made by the operator passed to the
    *similar* argument.

    :param maxsize: The maximum number of individual to keep in the hall of
                    fame.
    :param similar: An equivalence operator between two individuals, optional.
                    It defaults to operator :func:`operator.eq`.

    The class :class:`HallOfFame` provides an interface similar to a list
    (without being one completely). It is possible to retrieve its length, to
    iterate on it forward and backward and to get an item or a slice from it.
    """

    def __init__(self, maxsize, givenfit, similar=eq):
        self.maxsize = maxsize
        self.keys = list()
        self.items = list()
        self.similar = similar
        self.givenfit = givenfit

    def update(self, population):
        """Update the hall of fame with the *population* by replacing the
        worst individuals in it by the best individuals present in
        *population* (if they are better). The size of the hall of fame is
        kept constant.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        cpb = creator.FitnessMax()  # 可与其他Individual的Fitness对比的OBJECT
        setattr(cpb, 'wvalues', tuple((self.givenfit,)))

        def mul(a, b):
            "Same as a * b."
            return a * b

        def log_(A):
            temp = np.log(A)
            temp = wash(temp)
            return temp

        ##这里拆开evaluation function to three different functions
        def eval_(individual, matrices, test_matrix):  # return with not abs()
            new_matrix = []
            func = toolbox.compile(expr=individual)
            # 因子矩阵
            new_matrix = func(matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5])
            temp = matrices[0].copy()
            temp3 = pd.DataFrame()

            if len(alpha_database) != 0:  # 如果alpha因子库不为空
                for i in range(len(alpha_database[0])):
                    # temp2 储存了每一代残差矩阵（2 on 1, 3 on 2 & 1）
                    temp2 = pd.DataFrame([lis.iloc[i] for lis in alpha_database])
                    temp3['beta'] = log_(barra_beta1.iloc[i])
                    for t in range(len(temp2)):
                        temp3['n{name}'.format(name=t)] = temp2.iloc[t, :]
                    # 残差使得所有因子在样本内线性无关(Note: case might be different out of sample)
                    result = sm.OLS(new_matrix.iloc[i], temp3).fit()
                    temp.iloc[i] = result.resid
            else:
                for i in range(len(new_matrix)):  # 若未有一个alpha因子被挖掘
                    result = sm.OLS(new_matrix.iloc[i], log_(barra_beta1.iloc[i])).fit()
                    temp.iloc[i] = result.resid
            temp = wash(temp)
            return temp

        def corr_(matrix):
            sizel = len(temp) * len(return_matrix.iloc[0])
            try:
                corr = pearsonr(temp.values.reshape(1, sizel)[0], return_matrix.fillna(0).values.reshape(1, sizel)[0])[
                    0]
            except:
                corr = 0
            if np.isnan(corr):
                corr = 0
            return corr

        def bool_(corr):
            if corr < 0:
                return False
            else:
                return True

        if len(self) == 0 and self.maxsize != 0 and len(population) > 0:
            # Working on an empty hall of fame is problematic for the
            # "for else"
            self.insert([population[0], 1])

        for ind in population:
            # Note when writing this into another py doc, the 'Data' outght to be changed
            if ind.fitness >= cpb:  # or len(self) < self.maxsize:
                # print('eventually got one great')
                # print(str(ind))

                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer[0]):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    temp = eval_(ind, Data, return_matrix)
                    score = corr_(temp)
                    boo = bool_(score)
                    print(str(ind))
                    print(score)
                    if abs(score) < self.givenfit:
                        break

                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    # print('time to append alphabase')
                    alpha_database.append(temp)  # adding alpha matrix to the database for neutrulization

                    if boo:
                        self.insert([ind, 1])  # 1 for positive corr
                    else:
                        self.insert([ind, -1])  # -1 for negative corr

    def insert(self, item):
        """Insert a new individual in the hall of fame using the
        :func:`~bisect.bisect_right` function. The inserted individual is
        inserted on the right side of an equal individual. Inserting a new
        individual in the hall of fame also preserve the hall of fame's order.
        This method **does not** check for the size of the hall of fame, in a
        way that inserting a new individual in a full hall of fame will not
        remove the worst individual to maintain a constant size.

        :param item: The individual with a fitness attribute to insert in the
                     hall of fame.
        """
        item = deepcopy(item)
        self.items.insert(len(self), item)

    def remove(self, index):
        """Remove the specified *index* from the hall of fame.

        :param index: An integer giving which item to remove.
        """
        del self.keys[len(self) - (index % len(self) + 1)]
        del self.items[index]

    def clear(self):
        """Clear the hall of fame."""
        del self.items[:]
        del self.keys[:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __reversed__(self):
        return reversed(self.items)

    def __str__(self):
        return str(self.items)


### define operators

def wash(A):
    temp = A.fillna(0)
    temp = temp.replace([np.inf, -np.inf], 0)
    return temp


def erase(A):
    return 0


# Define new functions
def safeDiv(left, right):
    temp = left / right
    temp = wash(temp)
    return temp


# Define rank function'
def rank(A):
    return A.rank(axis=1)


# Define delay function'
def delay(A, n):
    temp = A.shift(n)
    temp.fillna(0)
    return temp


def question(condition, A, B):
    inv_con = condition.applymap(lambda x: not x)
    if type(A) == pd.DataFrame:
        if type(B) == pd.DataFrame:
            return A[condition].fillna(0) + B[inv_con].fillna(0)
        else:
            return A[condition].fillna(B)
    else:
        if type(B) == pd.DataFrame:
            return B[inv_con].fillna(A)
        else:
            return (condition * 1).replace(0, np.nan).replace(1, A).fillna(B)


def max_(A, B):
    if all((A == B).values[0]):
        temp = A.copy()
        temp[:] = 0
        return temp
    else:
        return question(A > B, A, B)


def min_(A, B):
    if all((A == B).values[0]):
        temp = A.copy()
        temp[:] = 0
        return temp
    else:
        return question(A < B, A, B)


def log_(A):
    temp = np.log(A)
    temp = wash(temp)
    return temp


def sdv(A, n):
    temp = A.fillna(method='ffill').rolling(n).std()
    temp = temp.fillna(0)
    return temp


def corr(A, B, n):
    temp = A.fillna(method='ffill').rolling(n).corr(other=B.fillna(method='ffill'))
    temp = temp.fillna(0)
    return temp


def delta(A, n):
    return (A - A.shift(n)).fillna(0)  # 可以不fillna因为再下一个操作中会被fillna，但最后出来的结果还是fillna比较好，为了防止deap出错以及fitness function的出错


def abs_(A):
    return abs(A)


def prod(A, n):
    return (A.fillna(method='ffill').rolling(n).apply(lambda x: x.prod())).fillna(0)


def WMA(A, n):
    weight = np.array([0.9 ** (n - 1 - i) for i in range(n)])
    weight = weight / weight.sum()
    return (A.fillna(method='ffill').rolling(n).apply(lambda x: x.dot(weight), raw=True)).fillna(0)


def sign(A):
    return A.applymap(lambda x: 1 if x > 0 else -1 if x < 0 else 0)


def decaylinear(A, d):
    weight = np.arange(d) + 1
    weight = weight / weight.sum()
    # rolling.apply容易出现大量的缺失数据
    return (A.fillna(method='ffill').rolling(d).apply(lambda x: x.dot(weight), raw=True)).fillna(0)


def covariance(A, B, n):
    if type(B) == pd.DataFrame:
        return (A.fillna(method='ffill').rolling(n).cov(other=B.fillna(method='ffill'))).fillna(0)
    else:
        return (A.fillna(method='ffill').rolling(n).apply(
            lambda x: np.cov(x, B.fillna(method='ffill'))[0][1])).fillna(0)


def sum_(A, n):
    return (A.fillna(method='ffill').rolling(n).sum()).fillna(0)


def mean(A, n):
    return (A.fillna(method='ffill').rolling(n).mean()).fillna(0)


def tsrank(A, n):
    if n != 0:
        A = A.fillna(method='ffill').rolling(n).apply(lambda x: x.searchsorted(x[-1]) + 1)
        return A.fillna(0)
    else:
        return A


def tsmax(A, n):
    if n == 0:
        return A
    else:
        return (A.fillna(method='ffill').rolling(n).max()).fillna(0)


def argmax(A, n):
    if n != 0:
        return A.fillna(method='ffill').rolling(n).apply(lambda x: np.where(x == max(x))[0][0] + 1, raw=True)
    # raw = True to silence apply function warning
    else:
        return A


def argmin(A, n):
    if n != 0:
        return A.fillna(method='ffill').rolling(n).apply(lambda x: np.where(x == min(x))[0][0] + 1, raw=True)
    else:
        return A


def tsmin(A, n):
    if n == 0:
        return A
    else:
        return (A.fillna(method='ffill').rolling(n).min()).fillna(0)


def mean2(A, B):
    return (A + B) / 2


def mean3(A, B, C):
    return (A + B + C) / 3


def clear_by_cond(A, B, C):
    temp = A.copy()
    temp[A < B] = 0  # 可以优化
    temp[A >= B] = C
    return temp


def negate(A):
    return -1 * A

stock = pd.read_pickle('C:/Users/86773/Desktop/AStockDailyData.pickle')
timey = stock.index
timey = timey.to_frame().drop_duplicates()

timey = timey[2888:3889]

import matplotlib.pyplot as plt
from scipy.signal import lfilter

logg = []

for macroi in range(int((len(timey) - 1) / 100 - 1)):

    ### import data

    random.seed(109 + macroi)
    start = timey.iloc[100 * macroi, 0]
    end = timey.iloc[100 * (macroi + 1), 0]
    subset = stock[(stock.index > start) & (stock.index < end)]

    code = subset['WINDCODE']
    code = code.drop_duplicates()
    code = pd.DataFrame(code)
    code = code.WINDCODE.tolist()

    subsetcode = random.sample(code, 100)

    trainset = subset[subset.WINDCODE.isin(subsetcode)]

    opens = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'open': trainset['open'] * trainset['adj']})
    close = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'close': trainset['close'] * trainset['adj']})
    high = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'high': trainset['high'] * trainset['adj']})
    low = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'low': trainset['low'] * trainset['adj']})
    amt = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'amt': trainset['pre_close']})
    volume = pd.DataFrame({'WINDCODE': trainset['WINDCODE'], 'volume': trainset['volume']})

    opens = opens.pivot(columns='WINDCODE', values='open')
    close = close.pivot(columns='WINDCODE', values='close')
    high = high.pivot(columns='WINDCODE', values='high')
    low = low.pivot(columns='WINDCODE', values='low')
    amt = amt.pivot(columns='WINDCODE', values='amt')
    volume = volume.pivot(columns='WINDCODE', values='volume')

    opens = opens.fillna(0)
    close = close.fillna(0)
    high = high.fillna(0)
    low = low.fillna(0)
    amt = amt.fillna(0)
    volume = volume.fillna(0)

    barra_beta = pd.read_pickle('C:/Users/86773/Desktop/factors_pickled/DailyFactor_CNE5_BarraBeta.txt')
    barra_beta1 = barra_beta.iloc[:, barra_beta.columns.isin(subsetcode)][
        (barra_beta.index > str(timey.iloc[100 * (macroi), 0])[0:10].replace('-', '')) & (
                    barra_beta.index < str(timey.iloc[100 * (macroi + 1), 0])[0:10].replace('-', ''))].fillna(0)

    opens = opens.iloc[:, opens.columns.isin(barra_beta1.columns)]
    close = close.iloc[:, close.columns.isin(barra_beta1.columns)]
    high = high.iloc[:, high.columns.isin(barra_beta1.columns)]
    low = low.iloc[:, low.columns.isin(barra_beta1.columns)]
    amt = amt.iloc[:, amt.columns.isin(barra_beta1.columns)]
    volume = volume.iloc[:, volume.columns.isin(barra_beta1.columns)]

    Data = (opens, close, high, low, amt, volume)

    starting_dat_return = timey.iloc[100 * (macroi), 0]
    ending_dat_return = timey.iloc[100 * (macroi + 1) + 1, 0]
    subset2 = stock[(stock.index > starting_dat_return) & (stock.index < ending_dat_return)]
    trainset2 = subset2[subset2.WINDCODE.isin(subsetcode)]
    preclose = pd.DataFrame({'WINDCODE': trainset2['WINDCODE'], 'pre_close': trainset2['pre_close'] * trainset2['adj']})
    preclose = preclose.pivot(columns='WINDCODE', values='pre_close')
    return_matrix = (preclose - preclose.shift(1)) / preclose.shift(1)
    return_matrix = return_matrix[
                        (return_matrix.index > starting_dat_return) & (return_matrix.index < ending_dat_return)][1:]
    return_matrix = return_matrix * 100  # 数值太小无法在除了covariance之后方式拟合
    return_matrix = wash(return_matrix)

    pset = []
    pset = gp.PrimitiveSetTyped("MAIN",
                                [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                pd.DataFrame)
    pset.addPrimitive(operator.add, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(operator.sub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(operator.mul, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(erase, [pd.DataFrame],
                      int)  # DEAP package automatically assumes that int object is empty if there is no addPrimitive output of 'int'; addTerminal gives an empty set for pset.primitives
    pset.addPrimitive(safeDiv, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(sign, [pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(rank, [pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(delay, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(delta, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(max_, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(min_, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(sdv, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(corr, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(abs_, [pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(prod, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(WMA, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(decaylinear, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(covariance, [pd.DataFrame, pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(sum_, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(mean, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(log_, [pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(tsrank, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(tsmax, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(tsmin, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(argmax, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(argmin, [pd.DataFrame, int], pd.DataFrame)
    pset.addPrimitive(mean2, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(mean3, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(clear_by_cond, [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame)
    pset.addPrimitive(negate, [pd.DataFrame], pd.DataFrame)
    pset.renameArguments(ARG0='open', ARG1='close', ARG2='high', ARG3='low', ARG4='amt', ARG5='volume')

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    alterbox = base.Toolbox()
    alterbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    alterbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)


    def evalSymbReg(individual, matrices, test_matrix):
        new_matrix = []
        # coe = 1 # set penalty coefficent
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # print(str(individual))
        # Evaluate the mean squared error between the expression
        # print(str(individual))
        temp = matrices[0].copy()
        # print('#{name}'.format(name = len(alpha_database)))
        # 惩罚arg 或 ts为开头的树
        if (str(individual)[0:3] == 'arg') or (str(individual)[0:2] == 'ts') or (str(individual)[0:4] == 'rank'):
            randomind = alterbox.individual()
            # print('non{name}'.format(name = str(individual)))
            return 0,
        # if ('arg' in str(individual)) or ('ts' in str(individual)):
        #    ts_arg = True
        # else:
        #    ts_arg = False
        #####
        #####
        if len(alpha_database) > 0:
            if (individual[0] in (older[0][0] for older in hof[1:len(hof)])):
                # 惩罚相同的树根的公式树
                #
                randomind = alterbox.individual()
                if (type(individual[0]) == gp.Terminal) or (type(randomind[0]) == gp.Terminal):
                    return 0,
                if individual[0].args == randomind[0].args:
                    # print('penalize here where {before} has the same root as alpha factor {alpha}'.format(before = str(individual),alpha = str(hof[1][0])))
                    individual[0] = randomind[0]  # 改树第一根
                    # print('it is changed randomly to be {after}'.format(after = str(individual)))
                # else:
            # s         print('改不了哦')
            func = toolbox.compile(expr=individual)

        new_matrix = func(matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5])
        temp3 = pd.DataFrame()
        # print(temp3)
        # print(temp)
        if len(alpha_database) != 0:
            for i in range(len(alpha_database[0])):
                temp2 = pd.DataFrame([lis.iloc[i] for lis in alpha_database])
                temp3['beta'] = log_(barra_beta1.iloc[i])
                for t in range(len(temp2)):
                    temp3['n{name}'.format(name=t)] = temp2.iloc[t, :]
                result = sm.OLS(new_matrix.iloc[i], temp3).fit()
                temp.iloc[i] = result.resid
        else:
            for i in range(len(new_matrix)):
                result = sm.OLS(new_matrix.iloc[i], log_(barra_beta1.iloc[i])).fit()
                temp.iloc[i] = result.resid
        sizel = len(temp) * len(temp.iloc[0])
        temp = wash(temp)
        try:
            corr = pearsonr(temp.values.reshape(1, sizel)[0], return_matrix.fillna(0).values.reshape(1, sizel)[0])[0]
        except:
            corr = 0
        if np.isnan(corr):
            corr = 0
        # if ts_arg:
        #    corr = corr * 1.5 #奖励ts or arg内树
        # print(corr)
        if len(alpha_database) == 0:
            if abs(corr) in check_dup_0:
                return 0,
            # print('@@{n}'.format(n=corr))
            check_dup_0.append(abs(corr))  # 第一代因子做严格化
            return abs(corr),
        else:
            if abs(corr) in check_dup_0:
                return 0,
            # print(corr)
            check_dup_1.append(abs(corr))
            return abs(corr),


    toolbox.register("evaluate", evalSymbReg, matrices=Data, test_matrix=return_matrix)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate",
                     gp.cxOnePoint)  # 别的mutation方法可以用deap.tools里的包，但是tools里的mate functions需要参照gp.cxonepoint修改，现在会出现bug
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=21))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=21))

    ###run the model
    popsize = 100
    givenfit = 0.3
    random.seed(1032)
    pset.addEphemeralConstant("cpcdcpcpcp: #{name}".format(name=macroi), lambda: random.randint(1, 10),
                              int)  # Remember to change the name each time
    warnings.filterwarnings("ignore")
    check_dup_0 = []
    check_dup_1 = []
    alpha_database = []
    pop = toolbox.population(n=popsize)
    hof = HallOfFame(1001, givenfit=givenfit)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    last_gen, logs = algorithms.eaSimple(pop, toolbox, 0.6, 0.4, 13  # Mate Prob, Mut Prob, & # of generations
                                         , stats=mstats,
                                         halloffame=hof, verbose=True)
    # print log
    # return pop, log, hof
    hof = hof[1:len(
        hof)]  # it is the deap coder decide that the first hof will always be a randomly bad tree from the first generation
    alpha_database = alpha_database[0:(len(alpha_database) - 1)]  # the last alpha matrix useless

    print(str(hof[0][0]))
    print(len(hof))
    logg.append(hof)

    # Testing
    testset = stock[
        (stock.index > timey.iloc[100 * (macroi + 1), 0]) & (stock.index < timey.iloc[100 * (macroi + 2), 0])]

    # Remember to distinguish the variables with the trainset

    opens = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'open': testset['open'] * testset['adj']})
    close = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'close': testset['close'] * testset['adj']})
    high = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'high': testset['high'] * testset['adj']})
    low = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'low': testset['low'] * testset['adj']})
    amt = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'amt': testset['pre_close']})
    volume = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'volume': testset['volume']})

    opens = opens.pivot(columns='WINDCODE', values='open')
    close = close.pivot(columns='WINDCODE', values='close')
    high = high.pivot(columns='WINDCODE', values='high')
    low = low.pivot(columns='WINDCODE', values='low')
    amt = amt.pivot(columns='WINDCODE', values='amt')
    volume = volume.pivot(columns='WINDCODE', values='volume')

    preclose = pd.DataFrame({'WINDCODE': testset['WINDCODE'], 'pre_close': testset['pre_close'] * testset['adj']})
    preclose = preclose.pivot(columns='WINDCODE', values='pre_close')
    return_matrix = (preclose - preclose.shift(1)) / preclose.shift(1)
    return_matrix = wash(return_matrix)


    # print(return_matrix)

    def testing_factor(individual, matrices, index):
        new_matrix = []
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual[0])
        # print(str(individual))
        # Evaluate the mean squared error between the expression
        # print(str(individual))
        new_matrix = func(matrices[0], matrices[1], matrices[2], matrices[3], matrices[4], matrices[5])

        temp = new_matrix.copy()
        temp3 = pd.DataFrame()

        if index == 1:
            for i in range(len(alpha_database_t[0])):
                temp2 = pd.DataFrame([lis.iloc[i] for lis in [alpha_database_t[0]]])
                temp3['beta'] = log_(barra_beta2.iloc[i])
                for t in range(len(temp2)):
                    temp3['n{name}'.format(name=t)] = temp2.iloc[t, :]
                # temp3 = wash(temp3)
                # print(temp3)
                result = sm.OLS(new_matrix.iloc[i], temp3).fit()
                temp.iloc[i] = result.resid
            alpha_database_t.append(wash(temp))

        if index > 1:
            for i in range(len(alpha_database_t[0])):
                temp2 = pd.DataFrame([lis.iloc[i] for lis in alpha_database_t[0:index]])
                temp3['beta'] = log_(barra_beta2.iloc[i])
                for t in range(len(temp2)):
                    temp3['n{name}'.format(name=t)] = temp2.iloc[t, :]
                temp3 = wash(temp3)
                result = sm.OLS(new_matrix.iloc[i], temp3).fit()
                temp.iloc[i] = result.resid
            alpha_database_t.append(wash(temp))

        if index == 0:
            print(new_matrix)
            for i in range(len(new_matrix)):
                result = sm.OLS(new_matrix.iloc[i], log_(barra_beta2.iloc[i])).fit()
                temp.iloc[i] = result.resid
            alpha_database_t.append(wash(temp))

        temp = wash(temp) * individual[1]
        return temp


    BS = return_matrix.iloc[0].copy()  # BOND SELECTION
    BS = BS * 0
    alpha_database_t = []  # new alpha data base
    barra_beta2 = barra_beta.iloc[:, barra_beta.columns.isin(opens.columns)][
        (barra_beta.index > str(timey.iloc[100 * (macroi + 1), 0])[0:10].replace('-', '')) & (
                    barra_beta.index < str(timey.iloc[100 * (macroi + 2), 0])[0:10].replace('-', ''))].fillna(0)
    barra_beta2 = pd.DataFrame.sort_index(barra_beta2, axis=1)  # 排序
    # 找共集
    opens = opens.iloc[:, opens.columns.isin(barra_beta2.columns)].fillna(0)
    close = close.iloc[:, close.columns.isin(barra_beta2.columns)].fillna(0)
    high = high.iloc[:, high.columns.isin(barra_beta2.columns)].fillna(0)
    low = low.iloc[:, low.columns.isin(barra_beta2.columns)].fillna(0)
    amt = amt.iloc[:, amt.columns.isin(barra_beta2.columns)].fillna(0)
    volume = volume.iloc[:, volume.columns.isin(barra_beta2.columns)].fillna(0)

    return_matrix = return_matrix.iloc[:, return_matrix.columns.isin(barra_beta2.columns)].fillna(0)  # 找共集
    testingData = (opens, close, high, low, amt, volume)
    # print(return_matrix)
    for i in range(len(hof)):
        print(str(hof[i][0]))
        temp = testing_factor(hof[i], testingData, i)
        # print (temp)
        BS = BS + temp.sum(axis=0)

    return_matrix = return_matrix.fillna(0) + 1

    BS = BS.to_frame()

    factor_value = BS.copy()

    factor_value['percentile'] = factor_value.rank(pct=True)

    i10 = factor_value[factor_value['percentile'] <= 0.1].index
    i9 = factor_value[(factor_value['percentile'] <= 0.2) & (factor_value['percentile'] > 0.1)].index
    i8 = factor_value[(factor_value['percentile'] <= 0.3) & (factor_value['percentile'] > 0.2)].index
    i7 = factor_value[(factor_value['percentile'] <= 0.4) & (factor_value['percentile'] > 0.3)].index
    i6 = factor_value[(factor_value['percentile'] <= 0.5) & (factor_value['percentile'] > 0.4)].index
    i5 = factor_value[(factor_value['percentile'] <= 0.6) & (factor_value['percentile'] > 0.5)].index
    i4 = factor_value[(factor_value['percentile'] <= 0.7) & (factor_value['percentile'] > 0.6)].index
    i3 = factor_value[(factor_value['percentile'] <= 0.8) & (factor_value['percentile'] > 0.7)].index
    i2 = factor_value[(factor_value['percentile'] <= 0.9) & (factor_value['percentile'] > 0.8)].index
    i1 = factor_value[factor_value['percentile'] > 0.9].index
    if macroi == 0:
        print('get heres')
        tenth = return_matrix.iloc[:, return_matrix.columns.isin(i10)].mean(axis=1)[1:-1].cumprod()
        nineth = return_matrix.iloc[:, return_matrix.columns.isin(i9)].mean(axis=1)[1:-1].cumprod()
        eighth = return_matrix.iloc[:, return_matrix.columns.isin(i8)].mean(axis=1)[1:-1].cumprod()
        seventh = return_matrix.iloc[:, return_matrix.columns.isin(i7)].mean(axis=1)[1:-1].cumprod()
        sixth = return_matrix.iloc[:, return_matrix.columns.isin(i6)].mean(axis=1)[1:-1].cumprod()
        fifth = return_matrix.iloc[:, return_matrix.columns.isin(i5)].mean(axis=1)[1:-1].cumprod()
        forth = return_matrix.iloc[:, return_matrix.columns.isin(i4)].mean(axis=1)[1:-1].cumprod()
        third = return_matrix.iloc[:, return_matrix.columns.isin(i3)].mean(axis=1)[1:-1].cumprod()
        second = return_matrix.iloc[:, return_matrix.columns.isin(i2)].mean(axis=1)[1:-1].cumprod()
        first = return_matrix.iloc[:, return_matrix.columns.isin(i1)].mean(axis=1)[1:-1].cumprod()

        overall = return_matrix.mean(axis=1)[1:-1].cumprod()

    else:
        print('not zero ')
        tenth1 = return_matrix.iloc[:, return_matrix.columns.isin(i10)].mean(axis=1)[1:-1].cumprod() * tenth[-1]
        nineth1 = return_matrix.iloc[:, return_matrix.columns.isin(i9)].mean(axis=1)[1:-1].cumprod() * nineth[-1]
        eighth1 = return_matrix.iloc[:, return_matrix.columns.isin(i8)].mean(axis=1)[1:-1].cumprod() * eighth[-1]
        seventh1 = return_matrix.iloc[:, return_matrix.columns.isin(i7)].mean(axis=1)[1:-1].cumprod() * seventh[-1]
        sixth1 = return_matrix.iloc[:, return_matrix.columns.isin(i6)].mean(axis=1)[1:-1].cumprod() * sixth[-1]
        fifth1 = return_matrix.iloc[:, return_matrix.columns.isin(i5)].mean(axis=1)[1:-1].cumprod() * fifth[-1]
        forth1 = return_matrix.iloc[:, return_matrix.columns.isin(i4)].mean(axis=1)[1:-1].cumprod() * forth[-1]
        third1 = return_matrix.iloc[:, return_matrix.columns.isin(i3)].mean(axis=1)[1:-1].cumprod() * third[-1]
        second1 = return_matrix.iloc[:, return_matrix.columns.isin(i2)].mean(axis=1)[1:-1].cumprod() * second[-1]
        first1 = return_matrix.iloc[:, return_matrix.columns.isin(i1)].mean(axis=1)[1:-1].cumprod() * first[-1]
        overall1 = return_matrix.mean(axis=1)[1:-1].cumprod() * overall[-1]

        tenth = tenth.append(tenth1)
        nineth = nineth.append(nineth1)
        eighth = eighth.append(eighth1)
        seventh = seventh.append(seventh1)
        sixth = sixth.append(sixth1)
        fifth = fifth.append(fifth1)
        forth = forth.append(forth1)
        third = third.append(third1)
        second = second.append(second1)
        first = first.append(first1)
        overall = overall.append(overall1)
        # print(tenth1)
        # print(tenth)
    fig = plt.figure(figsize=(12, 3))

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.plot(tenth.index, tenth, 'red', fifth.index, nineth, 'orange',
              tenth.index, eighth, 'yellow', fifth.index, seventh, 'green', forth.index, forth, '#eeefff',
    fifth.index, sixth, 'blue', third.index, fifth, 'purple',
    first.index, third, 'black', tenth.index, second, 'pink', tenth.index, first, 'cyan', linewidth = 3, markersize = 1)

    plt.show()