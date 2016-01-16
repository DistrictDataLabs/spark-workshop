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

# Parse string into floats
def parse(l):
    return array([float(x) for x in l.split(";")])

if __name__ == '__main__':
    conf = SparkConf().setAppName("Kmeans")
    sc   = SparkContext(conf=conf)


    # Load and parse the data
    # read in raw data
    print("Reading Data" + "\n")
    red_wines = sc.textFile("../fixtures/winequality-red.csv")
    white_wines = sc.textFile("../fixtures/winequality-white.csv")
    
    # separate the data from the headers
    red_wines_data = red_wines.zipWithIndex().filter(lambda s: s[1]>0).map(lambda s: s[0])
    white_wines_data = white_wines.zipWithIndex().filter(lambda s: s[1]>0).map(lambda s: s[0])
    
    parsedData = red_wines_data.map(parse).union(white_wines_data.map(parse))


    
    # Build the model (cluster the data)
    clusters = KMeans.train(parsedData, 2, maxIterations=10,
                            runs=10, initializationMode="random")


    WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE) + "\n")   

    print("Cluster Centers " + "\n")
    print(str(clusters.centers)) 