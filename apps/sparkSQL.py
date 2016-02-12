from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *


def parsePoint(l):
    return [float(x) for x in l.split(';')]



if __name__ == '__main__':
	conf = SparkConf().setAppName("SparkSQL")
	sc   = SparkContext(conf=conf)
	sqlCtx = SQLContext(sc)

	red_wines = sc.textFile("../fixtures/winequality-red.csv")

	# separate data from headers
	red_wines_data = red_wines.zipWithIndex().filter(lambda s: s[1]>0).map(lambda s: s[0])


	# Now begin setting up a DataFrame.
	# Since we're reading in data from flat files 
	#(and not an existing databse), we'll need to define the data schema.

	# The schema is encoded in a string. This example demos how to build a schema programatically.
	schemaString = "fixed_acidity volatile_acidity citric_acidity residual_sugar chlorides free_sulfur total_sulfur density pH sulphates alcohol quality"
	fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
	schema = StructType(fields)

	# Now we create a DataFrame, which exposes a different API than RDDs.
	red_wines_df = sqlCtx.createDataFrame(red_wines_data.map(parsePoint), schema)

	print("Data Frame Columns: \n")
	red_wines_df.show()

	# We can now select data columns/fields by field_name
	print("Looking at the FixedAcidity column: \n")
	red_wines_df.select("fixed_acidity").show()

	# We have methods like groupBy(), count(), and orderBy()
	print("Looking at the histogram of the FreeSulfur variable: \n")
	red_wines_df.groupBy("free_sulfur").count().orderBy("free_sulfur").show()

	## We can use SQL syntax
	# Use the DataFrame to instantiate a corresponding "table"
	red_wines_df.registerTempTable("wine_table")
	# Then we can do whatever SQL we want
	print("We can create a database table and use SQL syntax")
	sqlCtx.sql('select count(*) as NumGreaterThanFour from wine_table where free_sulfur > 4').show()