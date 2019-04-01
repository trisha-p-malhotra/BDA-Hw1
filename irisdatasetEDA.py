"""
__author__ = "Trisha P Malhotra (tpm6421)"

__desc__ = "Big Data Analytics - 720.02, homework 1"
    Qn.3)
    In class we discussed visualization using the iris dataset.
    In this exercise, you will compute the distancesbetween data points for the iris data set (iris.data).
     Specifically, you will write some code to
         a.read the iris data and form a data matrix X of size 150x4
         b.Normalize X  as X', where μjis the mean of attributejand jis the standard deviation of attribute j.
         c.Compute the distance matrix between pairs of data in X' using the following metrics:
            i.Euclideandistance
            ii.Cosine distance = 1-Cosine_similarity
            iii.Mahalanobis Distance
         d.Plot the above distance matrices (In python, you can use the matplotlib package and matshow to plot a matrix)
         e.Using the proximity measureyou developed in 2(b) computethe proximitybetween databelonging to the same class.
         Express the proximity as a symmetric 3x3 matrix(for distances between the threeclasses).
         Do this using allthree distance metrics in(c).Discuss your results.

"""

from sklearn import datasets
from sklearn import preprocessing
# for normalizing
from sklearn.metrics.pairwise import euclidean_distances
# for c.i
from sklearn.metrics.pairwise import cosine_distances
# for c.ii
from sklearn.metrics.pairwise import pairwise_distances
# for c.iii
import matplotlib.pyplot as plt
# for d


"""
a. Reading the dataset and loading it into matrix X
   of shape 150x4
"""
iris = datasets.load_iris()
X = iris.data
print("Iris dataset loaded into matrix X of shape : " + str(X.shape))      # (150, 4)



"""
b. Normalization: Converts all values, to fit between range 0 - 1
"""
# norm_X = X'
norm_X = preprocessing.normalize(X)
#print(norm_X)
print("X is now normalized between range 0 - 1")



"""
c.i Now, finding euclidean distance between rows of norm_X
    Formula used in this function is : dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
"""
euclidean_matrix = euclidean_distances(norm_X, norm_X)
#print(euclidean_matrix)
print("Euclidean matrix calculated for normalized X")
plt.matshow(euclidean_matrix)
plt.title("Euclidean distance")
plt.show()



"""
# extra work : eu-distance of X from origin
eu_origin_normX = euclidean_distances(norm_X, [[0,0,0,0]])
eu_origin_X = euclidean_distances(X, [[0,0,0,0]])
print(eu_origin_normX)
print(eu_origin_X)
"""

"""
c.ii Cosine distance = 1-Cosine_similarity
"""

cosine_dist = cosine_distances(norm_X,norm_X)
#print(cosine_dist)
print("Cosine matrix calculated for normalized X")
plt.matshow(cosine_dist)
plt.title("Cosine distance")
plt.show()



"""
from sklearn.neighbors import DistanceMetric


maha1 = DistanceMetric.get_metric('mahalanobis')
maha = maha1.pairwise(norm_X, 'mahalanobis', 'V')
print("mahahaaaaa")
print(maha)
"""


#================================================
"""
3.e, three separate classes based arrays:
"""
# SETOSA
class_setosa = iris.data[range(0,50)]
norm_setosa = preprocessing.normalize(class_setosa)
# euclidean
euclidean_setosa = euclidean_distances(norm_setosa, norm_setosa)
plt.matshow(euclidean_setosa)
plt.title("Euclidean for class setosa")
plt.show()
# cosine
cosine_setosa = cosine_distances(norm_setosa,norm_setosa)
plt.matshow(cosine_setosa)
plt.title("Cosine for class setosa")
plt.show()
print("Euclidean and cosine distance matrices calculated for class Setosa")


# VERSICOLOR
class_versicolor = iris.data[range(50,100)]
norm_versicolor = preprocessing.normalize(class_versicolor)
euclidean_versicolor = euclidean_distances(class_versicolor, class_versicolor)
plt.matshow(euclidean_versicolor)
plt.title("Euclidean for class_versicolor")
plt.show()
# cosine
cosine_versicolor = cosine_distances(norm_versicolor,norm_versicolor)
plt.matshow(cosine_versicolor)
plt.title("Cosine for class versicolor")
plt.show()
print("Euclidean and cosine distance matrices calculated for class Versicolor")



# VIRGINICA
class_virginica = iris.data[range(100,150)]
norm_virginica = preprocessing.normalize(class_virginica)
# euclidean
euclidean_virginica = euclidean_distances(norm_virginica, norm_virginica)
plt.matshow(euclidean_virginica)
plt.title("Euclidean for virginica")
plt.show()
# cosine
cosine_virginica = cosine_distances(norm_virginica,norm_virginica)
plt.matshow(cosine_virginica)
plt.title("Cosine for class virginica")
plt.show()
print("Euclidean and cosine distance matrices calculated for class Virginica")

