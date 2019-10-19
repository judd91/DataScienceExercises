# Data Science Exercises

Collection of simple data science challenges.

## Classification Problem - Spam Classifier

The goal is to create a machine learning model to classify spam and not spam text-messages. 

The dataset of labeled text messages is from the UCI Machine Learning Repository which is randomly split into a training- and test-dataset.

Apache Spark ML library for scala is used to create a pipeline. The classification method applied to the training dataset is Naïve Bayes. In order to evaluate the classifier performance, MulticlassClassificationEvaluator and CrossValidator functions are used.

Accuracy is calculated to get the performance of the model. In this case, the accuracy output is 0,982301 so it is a positive result. It means this pipeline classifier has a high performance and there is an evidence for taking the model trusty.

## Linear Regression - Traffic volume

This script try to find the relationship between the time of the day and traffic volume when is not holiday day. The programming language used is python.

The data is cleaned and transformed in order to get better results.

It is possible to obtain a prediction of the traffic volume giving an specific hour. Coefficients, independent term, mean squared error and variance score are also calculated in order to get a better accuracy.

Traffic data from MN Department of Transportation Weather data from OpenWeatherMap. Dataset used can be downloaded from UCI (http://archive.ics.uci.edu)

## Clustering - K-means 

K-means of Spark’s machine learning library(MLlib) is used on the MNIST dataset of handwritten digits (http://yann.lecun.com/exdb/mnist/) in order to get the cluster centroids. It is a scala script. The output is saved in a .csv file. Alfterwards, the centroids are visualiced in a python file using matplotlib and numpy.

## Image Descriptor

Python language and OpenCV library are used in order to obtain basic descriptors from an image.

Color Detector: k-means function is used to build a color histogram from an RGB image by clustering the pixel intensities. k-means is a clustering algorithm to partition n data points into k clusters. Each of them will be assigned to a cluster with the nearest mean.
![alt text](https://github.com/judd91/DataScienceExercises/blob/master/ImageDescriptor/Images%20Result/ColorDesc.PNG)

Shape Detector: The goal is to find black shapes in the image. *cv2.findContours()* is used to detect the contours of the figure.
![alt text](https://github.com/judd91/DataScienceExercises/blob/master/ImageDescriptor/Images%20Result/shape.PNG)




