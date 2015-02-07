import csv

from StringIO import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

def split(line):
    """
    Operator function for splitting a line on a delimiter.
    """
    reader = csv.reader(StringIO(line))
    return reader.next()

def main(sc, sqlc):

    rows = sc.textFile("fixtures/shopping/customers.csv").map(split)
    customers = rows.map(lambda c: Row(id=int(c[0]), name=c[1], state=c[6]))

    # Infer the schema and register the SchemaRDD
    schema = sqlc.inferSchema(customers)
    schema.registerTempTable("customers")

    maryland = sqlc.sql("SELECT name FROM customers WHERE state = 'Maryland'")
    print maryland.count()

if __name__ == '__main__':
    conf = SparkConf().setAppName("Query Customers")
    sc   = SparkContext(conf=conf)
    sqlc = SQLContext(sc)
    main(sc, sqlc)
