import sys
import random

from operator import add
from pyspark import SparkConf, SparkContext

def estimate(idx):
    x = random.random() * 2 - 1
    y = random.random() * 2 - 1
    return 1 if (x*x + y*y < 1) else 0

def main(sc, *args):
    slices = int(args[0]) if len(args) > 0 else 2
    N = 100000 * slices

    count  = sc.parallelize(xrange(N), slices).map(estimate)
    count  = count.reduce(add)

    print "Pi is roughly %0.5f" % (4.0 * count / N)
    sc.stop()

if __name__ == '__main__':
    conf = SparkConf().setAppName("Estimate Pi")
    sc   = SparkContext(conf=conf)
    main(sc, *sys.argv[1:])
