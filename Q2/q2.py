#
#CLEAN UP CODE AND ADD COMMENTS 
#MENTION IN README WHERE TO CHANGE CODE
#
#read input image as a matrix
#change from color to grayscale

 
from cmath import pi
import cv2
from matplotlib import image
import numpy as np
import math
import matplotlib.pyplot as plt

img1 = cv2.imread('source code\Images\cat2.jpg',0)
img2 = cv2.imread('source code\Images\img0.jpg',0)
img3 = cv2.imread('source code\Images\littledog.jpg',0)
# cv2.imshow('image',img1)
# cv2.waitKey(0)
#plt.imshow(img, cmap='gray')
# print(img1.shape, img2.shape, img3.shape)
 
#------do a gaussian blur---------

# calculate kernel
#this function takes in sigma, calculates kernel dimensions then computes the gaussian kernel
def createKernel(sigma):
    #kernel size
    k = 2 * math.ceil(3*sigma) + 1
    kernel = np.zeros((k,k), dtype=float)
    mean = k//2
    sum = 0
    for x in range(0, k):
        for y in range (0 , k):
            brackets = -0.5*(((x-mean)**2 + (y-mean)**2)/(sigma**2))
            kernel[x][y] = math.exp(brackets)/2*math.pi*(sigma**2)
            sum += kernel[x][y]
    #normalizing kernel values by dividing by sum
    for x in range(0, k):
        for y in range (0 , k):
            kernel[x][y] /= sum
    return kernel

#------ do convolution ---
# this funciton takes in an image and a kernel and calculates the convolution
def calcConvolution(image,kernel):
    imgconv= np.zeros((image.shape[0], image.shape[1]), dtype=float)
    kernel_k = kernel.shape[0]
    img_x = image.shape[0]
    img_y = image.shape[1]
    k = kernel_k //2
    for i in range (0,img_x):
        for j in range (0,img_y):
            val = 0
            for u in range (-k , k+1):
                for v in range(-k , k+1):           
                #--implementation of wrapping
                #--conditions for wrapping
                    x = i - u
                    y = j - v
                    if x < 0 :
                        x = img_x - u -1
                    if y < 0 :
                        y = img_y - v -1
                    if x > img_x - 1 :
                        x = x - img_x - 1 
                    if y > img_y-1:
                        y = y - img_y - 1
                    val += kernel[u+k][v+k] * image[x][y]
            imgconv[i][j] = val
    return imgconv


def gaussianBlur(image, sigma):
    kernel = createKernel(sigma)
    conv = calcConvolution(image,kernel)
    return conv



#sobel edge detection in x and y direction

#--calculates the sobel in x and y then gets the magnitude
def magnitude(image):
    sobel_x = np.zeros((3,3), dtype=float)
    sobel_x[0][0] = 1
    sobel_x[0][1] = 0
    sobel_x[0][2] = -1
    sobel_x[1][0] = 2
    sobel_x[1][1] = 0
    sobel_x[1][2] = -2
    sobel_x[2][0] = 1
    sobel_x[2][1] = 0
    sobel_x[2][2] = -1
    sobel_y = np.transpose(sobel_x)
    img_x = calcConvolution(image,sobel_x)
    img_y = calcConvolution(image, sobel_y)
    x = image.shape[0]
    y = image.shape[1]
    img_mag = np.zeros((x,y), dtype=float)
    for i in range(x):
        for j in range(y):
            mag = math.sqrt(img_x[i][j]**2 + img_y[i][j]**2)
            img_mag[i][j] = mag
    return (img_mag,img_x,img_y)



def calcAngleDegrees(y, x):
    a = math.atan2(y, x) * 180 / math.pi
    if a < 0:
        a = a + 180
    if a >= 22.5 and a <= 67.5:
        a = 45
    elif a > 67.5 and a <=112.5:
        a = 90
    elif a > 112.5 and a <= 157.5:
        a = 135
    else :
        a = 0
    return a



def getCoordinates(i,j,sizex,sizey,u,v):
    x = i + u
    y = j + v
    if x < 0 :
        x = -1
    if y < 0 :
        y = -1
    if x > sizex - 1 :
        x = -1
    if y > sizey -1:
        y = -1
    return (x,y)

# uses the image magnitue, sobel x and y to calculate the non maxima supressed image
def supression(img_mag,img_x,img_y):
    k_x = img_mag.shape[0]
    k_y = img_mag.shape[1]
    x1 = y1 = x2 = y2 = 0
    new_img = np.zeros((k_x,k_y), dtype=float)
    for i in range(k_x):
        for j in range(k_y):
            atan = calcAngleDegrees(img_y[i][j], img_x[i][j])
            # 0,45,90,135
            if atan == 0:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,0,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,0,-1)
            elif atan == 45:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,-1)
            elif atan == 90:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,0)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,0)
            elif atan == 135:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,1)
            
            if x1 != -1 and x2 != -1 and y1 != -1 and y2 != -1:
                if img_mag[i][j] < img_mag[x1][y1] or  img_mag[i][j] < img_mag[x2][y2]:
                     new_img[i][j] = 0
                else :
                    new_img[i][j] = img_mag[i][j]
            else :
                new_img[i][j] = img_mag[i][j]
    return new_img

def threshold(thres,img):
    k = img.shape[0]
    for i in range(k):
        for j in range(k):
            if img[i][j] < thres:
                img[i][j] = 0
    return img

# to change the image and sigma change the input values to the gaussian blur
imageout = gaussianBlur(img2,2)
img_m,img_x,img_y = magnitude(imageout)
img_sup = supression(img_m,img_x,img_y)
img_thr = threshold(50 ,img_sup)

cv2.imwrite('source code\Edge detection\Blur.jpg',imageout.astype(np.uint8))
cv2.imwrite('source code\Edge detection\Magnitude.jpg',img_m.astype(np.uint8))
cv2.imwrite('source code\Edge detection\Suppresion.jpg',img_sup.astype(np.uint8))
cv2.imwrite('source code\Edge detection\Threshold.jpg',img_thr.astype(np.uint8))

cv2.imshow('Blur',imageout.astype(np.uint8))
cv2.imshow('Magnitude',img_m.astype(np.uint8))
cv2.imshow('Suppresion',img_sup.astype(np.uint8))
cv2.imshow('Threshold',img_thr.astype(np.uint8))
cv2.waitKey(0)