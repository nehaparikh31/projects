import numpy as np

from emdemo import normPDF, elementWiseAdd, weightedPDF, copyArray, closestCluster
from kmeans import parseLine, getTrueValue
from pyspark import SparkConf, SparkContext
from operator import add


class EM:
    def __init__(self):
        self.numClusters = 0
        self.theta = None
        self.mu = None
        self.sigma = None

    def assignLabels(self, sourceRDD, filename):
        parsedData = sourceRDD.map(parseLine)
        trueValue = sourceRDD.map(getTrueValue)
        predictedLabels = parsedData.map(
            lambda point: (point, weightedPDF(point, self.numClusters, self.theta, self.mu, self.sigma)))
        results = predictedLabels.collect()
        true = trueValue.collect()
        accuracy_count = 0
        accuracy = 0

        with open(filename, "w") as f:
            f.write("true\tpredicted\n")
            for i in range(len(results)):
                f.write(str(true[i]) + "\t" + str(results[i]) + "\n")
                if int(true[i]) == int(results[i]):
                    accuracy_count += 1
            accuracy = accuracy_count / len(results)
            if accuracy < 0.5:
                accuracy = 1 - accuracy
            f.write("accuracy is:" + str(accuracy))

        print("accuracy is:", accuracy)

    def EMClustering(self, sourceRDD, K, maxIteration):
        self.numClusters = K
        parsedData = sourceRDD.map(parseLine)
        trueValue = sourceRDD.map(getTrueValue)
        features = len(parsedData.take(1)[0])  # number of features of a data point
        print("number of features: ", features)
        samples = parsedData.count()  # number of data points
        print("number of data points: ", samples)

        # initialize three model parameters:
        theta = np.ones(K) / 2  # theta[0], theta[1] = 0.5
        mu = parsedData.takeSample(False, K, 1)  # mu[0], mu[1] for K = 2 are sample means, each is a vector
        sigma = np.zeros((K, features, features))
        for i in range(K):  # sigma[0] is co-variance matrix of data points in cluster 0
            sigma[i] = np.eye(features)  # sigma[1] is co-variance matrix of data points in cluster 1
        DELTA = np.eye(features) * 1e-7  # fixed diagonal matrix of small values, to avoid singular matrix
        for count in range(maxIteration):
            error = 0.0
            oldTheta = copyArray(theta)
            for i in range(K):
                if (np.linalg.det(sigma[i]) == 0):  # compute determinant of matrix
                    sigma[i] = sigma[i] + DELTA  # to avoid singular matrix
            thetaPDF = parsedData.map(lambda point: (point, weightedPDF(point, K, theta, mu, sigma)))

            resultsThetaPDF = thetaPDF.collect()
            print("thetaPDFstart")
            for i in range(len(resultsThetaPDF)):
                print(resultsThetaPDF[i])
            print("thetaPDFend")

            arrayOfSizeK0 = thetaPDF.map(lambda line: line[1][0]).collect()
            arrayOfSizeK1 = thetaPDF.map(lambda line: line[1][1]).collect()

            # <point, probability that point is in cluster0>
            pointProb0 = thetaPDF.map(lambda line: (line[0], line[1] / (line[1] + line[2])))
            pointProb1 = thetaPDF.map(lambda line: (line[0], line[2] / (line[1] + line[2])))

            for i in range(K):
                theta[i] = thetaPDF.map(lambda line: line[1][i]).reduce(add)
            theta = theta / samples
            print("th0 final value: ", theta[0])
            print("th1 final value: ", theta[1])

            for i in range(K):
                mu[i] = thetaPDF.map(lambda line: line[0] * line[1][i]).reduce(elementWiseAdd)
                mu[i] = mu[i] / (samples * theta[i])

            Sigma0 = pointProb0.map(lambda line: normPDF(line[0], line[1], self.mu[0])).reduce(elementWiseAdd)
            toSigma0 = Sigma0 / (samples * self.theta[0])

            Sigma1 = pointProb1.map(lambda line: normPDF(line[0], self.mu[1], line[1])).reduce(elementWiseAdd)
            toSigma1 = Sigma0 / (samples * self.theta[1])

            self.sigma[0] = np.reshape(toSigma0, (features, features))
            self.sigma[1] = np.reshape(toSigma1, (features, features))

            predictedLabels2 = parsedData.map(lambda point: closestCluster(weightedPDF(point, 2, theta, mu, sigma)))
            results2 = predictedLabels2.collect()
            true2 = trueValue.collect()
            accuracy_count = 0
            accuracy = 0
            outfilename = "EMresults.txt"
            with open(outfilename, "w") as fileVar:
                fileVar.write("true\tpredicted\n")
                for i in range(len(results2)):
                    fileVar.write(str(true2[i]) + "\t" + str(results2[i]) + "\n")
                    if int(true2[i]) == int(results2[i]):
                        accuracy_count += 1
                accuracy = accuracy_count / len(results2)
                if accuracy < 0.5:
                    accuracy = 1 - accuracy
                fileVar.write("accuracy is:" + str(accuracy))

            print("accuracy is:", accuracy)
            sc.stop()