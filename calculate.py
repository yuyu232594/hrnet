import numpy as np
import cv2 as cv
import random
import os
import csv
path_csv='/Users/wenyu/Desktop/demo/iris.csv'
# 灰度图像的均值和标准差
means=[]
stdss=[]

def get_images(path):
    images=[]
    with open(path) as f:
        reader=csv.reader(f)
        for row in reader:
            img,_=row
            images.append(img)
    return images

def get_mean_stdevs(images_list):
    length=len(images_list)
    for i in range(length):
        img_origin=cv.imread(images_list[i])
        img_gray=cv.cvtColor(img_origin,cv.COLOR_BGR2GRAY)
        img_gray=np.asarray(img_gray)
        # 这一步后面必须加一个.因为需要变成小数
        img_gray=img_gray.astype(np.float32)/255.
        img_gray_mean=np.mean(img_gray)
        img_gray_std=np.std(img_gray)
        means.append(img_gray_mean)
        stdss.append(img_gray_std)
    data_mean=np.mean(means)
    data_std=np.mean(stdss)
    return data_mean,data_std

    

if __name__ == '__main__':
    images_list=get_images(path_csv)
    mean,stds=get_mean_stdevs(images_list)
    print(mean,stds)
