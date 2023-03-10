import time
import numpy as np
import cv2

#Code for adaptive histogram
#Select kerenel value
kernel_val =100

kernel_size = int(kernel_val)
kernel_size_squared = int(kernel_size * kernel_size)


image = cv2.imread('/home/shubham/0000000000.png',0)
img = np.lib.pad(image,(kernel_size,kernel_size),'reflect')
img_size=img.shape
max_intensity = 255
final_img = np.zeros_like(img)

#Iterate through the pixels in the image
for i in range(0,img_size[0]-kernel_size):
    #print(i)
    for j in range(0,img_size[1]-kernel_size):
		#Extract a window from the image
        kernel = img[i:i+kernel_size,j:j+kernel_size]
		#Sort the extracted window pixels
        kernel_flat = np.sort(kernel.flatten())
        # Calculate the rank of the pixel
        rank = np.where(kernel_flat == img[i,j])[0][0]
		#Write the value of the new pixel to the other array
        final_img[i,j] = int((rank * max_intensity )/(kernel_size_squared))


cv2.imshow("Adaptive",final_img )
cv2.waitKey(0)




