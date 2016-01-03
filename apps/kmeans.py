## Spark Application for performing Kmeans clustering.

import csv

from numpy import array
from math import sqrt
from StringIO import StringIO

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


if __name__ == '__main__':
    conf = SparkConf().setAppName("Kmeans")
    sc   = SparkContext(conf=conf)


    # Load and parse the data
    data = sc.textFile("../fixtures/kmeans_data.txt")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

    
    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 2, maxIterations=10,
                            runs=10, initializationMode="random")


    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))    