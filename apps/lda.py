## Spark Application for performing Latent Dirichlet Allocation.


from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors


# Load and parse the data


if __name__ == '__main__':
    conf = SparkConf().setAppName("LDA")
    sc   = SparkContext(conf=conf)

    # Load and parse the data
    data = sc.textFile("../fixtures/sample_lda_data.txt")
    # data file found at https://raw.githubusercontent.com/apache/spark/master/data/mllib/sample_lda_data.txt
    parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))

    # Index documents with unique IDs
    corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
    
    # Cluster the documents into three topics using LDA
    ldaModel = LDA.train(corpus, k=3)

    # Output topics. Each is a distribution over words (matching word count vectors)
    print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
    topics = ldaModel.topicsMatrix()
    for topic in range(3):
        print("Topic " + str(topic) + ":")
        for word in range(0, ldaModel.vocabSize()):
            print(" " + str(topics[word][topic]))
