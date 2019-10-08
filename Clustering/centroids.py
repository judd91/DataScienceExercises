import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im

with open("centroids.csv", 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    i = 1
    for row in csvreader:
        a = np.array(row).reshape(28,28)
        im.imsave('im'+ str(i)+'.jpg', a)
        i = i + 1




