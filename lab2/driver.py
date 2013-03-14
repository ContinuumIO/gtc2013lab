import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer
#from matplotlib import pyplot

from config import *
def driver(pricer):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    ts = timer()
    pricer(paths, DT, InterestRate, Volatility)
    te = timer()

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print 'StockPrice', np.mean(ST)
    print 'StandardError', np.std(ST) / sqrt(NumPath)
    print 'PaidOff', np.mean(PaidOff)
    OptionPrice = np.mean(PaidOff) * exp(-InterestRate * Maturity)
    print 'OptionPrice', OptionPrice

    print

    NumCompute = NumPath * NumStep
    print '%.2f' % (NumCompute / (te - ts) / 1e6), 'MStep per second'
    print '%fs' % (te - ts)


#    for i in xrange(30):
#        pyplot.plot(paths[i])
#    pyplot.show()

