## Spark Application for performing LDA with text from War and Peace.


from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
import numpy as np


def vectorizer(sentence,vocab):
	"""
	Converts a text sentence into a vector representation in the vocabulary vector space.
	"""
	vocabulary = vocab.value
	vec = np.zeros(len(vocabulary))
	for word in sentence.split(" "):
		if word in vocabulary:
			vec[vocabulary.index(word)] += 1
	return Vectors.dense(vec)

if __name__ == '__main__':
	# Set up Spark
	conf = SparkConf().setAppName("LDA")
	sc   = SparkContext(conf=conf)

	# Read in data, filter out empty lines   
	tolstoy = sc.textFile("../fixtures/tolstoy.txt")
	sentences = tolstoy.filter(lambda s: len(s)>0)

	# We have a fair amount of data wrangling to do to get things into the right format for Spark's LDA. 

	# First, we're going to identify the top words in the corpus and only keep track of those words. 
	# Those top words will form our vocabulary.
	word_counts = sentences.flatMap(lambda s:  s.split(" ")).map(lambda w: (w.lower(),1)).reduceByKey(lambda a,b : a+b)
	top_words = word_counts.takeOrdered(500,key=lambda (w,c):-c)
	vocabulary = [str(k) for (k,v) in top_words]

	# We also want a Broadcast version of the vocabulary list.
	br_vocabulary = sc.broadcast(vocabulary)

	# Next, we need to convert the raw text sentences into a dense-vector representation.
	dense_vectors = sentences.map(lambda s: vectorizer(s,br_vocabulary))

	# Finally, we create our corpus by giving each sentence an ID.
	corpus = dense_vectors.zipWithIndex().map(lambda (v,i): [i, v]) 

	# Now we can train an LDA model on our data.
	lda_model = LDA.train(corpus, k =3, maxIterations=20)


	# Output topics. For each topic, print out the top words contributing to that topic.
	print("Learned topics (as distributions over vocab of " + str(lda_model.vocabSize()) + " words):")
	topics = lda_model.topicsMatrix()
	for topic in range(topics.shape[1]):
		print("Topic " + str(topic) + ":")    
		topic_word_counts = sorted(zip(vocabulary,lda_model.topicsMatrix()[:,topic]), key = lambda (w,c):-c )
		top_words = [w for (w,c) in topic_word_counts[:30]]
		print top_words