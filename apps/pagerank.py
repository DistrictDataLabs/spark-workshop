import csv

from numpy import array
from StringIO import StringIO

from operator import add
from pyspark import SparkConf, SparkContext

def contribution(authors, rank):
    """Calculates URL contributions to the rank of other URLs."""
    count = len(authors)
    for author in authors:
        yield (author, rank / count)

def split(line):
    """
    Operator function for splitting a line on a delimiter.
    """
    reader = csv.reader(StringIO(line))
    return tuple(reader.next())

if __name__ == '__main__':
    sc = SparkContext("local[*]", "PageRank")

    # Loads in input file. It should be in format of:
    # AuthorID, AuthorID
    lines = sc.textFile('fixtures/DBLP/coauthors.txt')

    # Loads all authors from input file and initialize their neighbors.
    links = lines.map(lambda line: split(line)).distinct().groupByKey().cache()

    # Loads all authors with other authors(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda (author, neighbors): (author, 1.0))


    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in xrange(10):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(lambda (author, (authors, rank)):
            contribution(authors, rank))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)



    # Collects all URL ranks and dump them to console.
    for (link, rank) in ranks.collect():
        print "%s has rank: %s." % (link, rank)
