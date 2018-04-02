# -*- coding: utf-8 -*-



from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
import nltk


from sklearn import cluster
from sklearn import metrics

# training data

sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
			['this', 'is',  'another', 'book'],
			['one', 'more', 'book'],
			['this', 'is', 'the', 'new', 'post'],
          ['this', 'is', 'about', 'machine', 'learning', 'post'],  
			['and', 'this', 'is', 'the', 'last', 'post']]


# training model
model = Word2Vec(sentences, min_count=1)

# get vector data
X = model[model.vocab]
print (X)

print (model.similarity('this', 'is'))

print (model.similarity('post', 'book'))

print (model.most_similar(positive=['machine'], negative=[], topn=2))

print (model['the'])

print (list(model.vocab))

print (len(list(model.vocab)))




NUM_CLUSTERS=3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)

words = list(model.vocab)
for i, word in enumerate(words):  
    print (word + ":" + str(assigned_clusters[i]))



kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)

print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print ("Silhouette_score: ")
print (silhouette_score)
