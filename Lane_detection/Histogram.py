import numpy as np
import cv2
from matplotlib import pyplot as plt, cm, colors

path = "/home/shubham/0000000000.png"
img = cv2.imread(path,0)

img = np.asarray(img)
flat=img.flatten()

# create our own histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)
    
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    
    # return our final result
    return histogram
hist = get_histogram(flat, 256)
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)   


cs = cumsum(hist)
nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()
cs = nj / N
cs = cs.astype('uint8')
img_new = cs[flat]
img_new = np.reshape(img_new, img.shape)

cv2.imshow("histogram_eq", img_new)
cv2.waitKey(0)





