import numpy as np
from matplotlib._cm_listed import data
from pyspark import SparkContext, rdd
from matplotlib import pyplot as plt
from EM import EMClustering

def getTrueValue(line):
    y = np.array([float(x) for x in line.split(',')])
    return y[-1] # return the last element (index -1 means last index)


def parseLine(line):
    y = np.array([float(x) for x in line.split(',')])
    return y[:-1] # get elements from index 0, stopping before last index


def elementWiseAdd(list1, list2):
    return [a + b for a, b in zip(list1, list2)]


def normPDF(point, mu, sigma):
    return multivariate_normal(mu, sigma).pdf(point)

def weightedPDF(point, K, theta, mu, sigma):
    pdf = np.zeros(K) # initialize array
    for i in range(K):
        pdf[i] = theta[i] * normPDF(point, mu[i], sigma[i])
    return pdf/sum(pdf)

def copyArray(list):

    newArray = np.zeros(len(list))
    for i in range(len(list)):
        newArray[i] = list[i]
    return newArray

def sumError(array0, array1):
    return sum([np.math.fabs(x) for x in (array0 - array1)])


def closestCluster(list):
    maxProb = 0
    for i in range(len(list)):
        if (list[i] > maxProb):
            maxProb = list[i]
            clusterlabel = i
    return clusterlabel


def main():
    sc = SparkContext(master="local", appName="EM")
    try:
        csv = sc.textFile("kmeans_data.csv")
    except IOError:
        print('No such file')
        exit(1)
        K = 2
        maxIteration = 5
        myEM = EM()
        myEM.EMClustering(rdd, K, maxIteration)
        outfilename = "EMresult.txt"
        myEM.assignLabels(rdd, outfilename)

        sc.stop()

if __name__ == "__main__":
    main()
