# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:31:57 2019

@author: wuzhe
"""
import cv2
from math import floor
import numpy as np
import dhash
from PIL import Image


"""
author: zhenyu wu
time: 2019/12/04 16:03
function: 均值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 均值哈希指纹计算结果
"""
def HashValue(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = floor(img[i,j]/4)
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp


"""
author: zhenyu wu
time: 2019/12/04 16:04
function: 根据均值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def Hash(img1, img2):
    img1 = HashValue(img1)
    img2 = HashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    if result<=5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 感知哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 感知哈希指纹计算结果
"""
def pHashValue(img):
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img = cv2.dct(img)
    img = img[:8, :8]
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp


"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 根据感知哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def pHash(img1, img2):
    img1 = pHashValue(img1)
    img2 = pHashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    if result<=5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/09 09:14
function: 差值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 差值哈希指纹计算结果
"""
def DHashValue(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img2 = []
    for i in range(8):
        img2.append(np.array(img[:,i])-np.array(img[:,i+1]))
    img2 = np.mat(img2).T
    img2[img2 >= 0] = 1
    img2[img2 < 0] = 0
    img2 = img2.reshape((1,64))
    return img2


"""
author: zhenyu wu
time: 2019/12/09 09:13
function: 根据差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def DHash(img1, img2):
    img1 = DHashValue(img1)
    img2 = DHashValue(img2)
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    if result<=5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/09 09:37
function: 根据包中的差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def dHash_use_package(img1, img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    row1, col1 = dhash.dhash_row_col(image1)
    row2, col2 = dhash.dhash_row_col(image2)
    a1 = int(dhash.format_hex(row1, col1), 16)
    a2 = int(dhash.format_hex(row2, col2), 16)
    result = dhash.get_num_bits_different(a1, a2)
    if result<=5:
        print('Same Picture')
    return result
    
    
if __name__ == '__main__':
    img1_path = 'picture_org.jpg'
    img2_path = 'picture_w.jpg'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    hash_result = Hash(img1, img2)
    print('Hash Hanming Distance: %d' % (hash_result))
    phash_result = pHash(img1, img2)
    print('pHash Hanming Distance: %d' % (phash_result))
    dhash_result = DHash(img1, img2)
    print('DHash Hanming Distance: %d' % (dhash_result))
    dhash_result_pkg =dHash_use_package(img1_path, img2_path)
    print('DHash Hanming Package Distance: %d' % (dhash_result_pkg))
    
