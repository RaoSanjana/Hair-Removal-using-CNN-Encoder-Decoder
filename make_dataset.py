# modified to make masks to go with hair and hairless image pairs

import os
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import glob
from os.path import basename

BASE_PATH = 'jpeg-melanoma-512x512/train/'

def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image

def get_mask(im):
    kernel = cv2.getStructuringElement(1,(17,17))
    grayScale = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret,threshold = cv2.threshold(blackhat,30,255,cv2.THRESH_BINARY)
    mask = threshold.astype(np.float32)          #cv2.normalize(threshold, threshold, 0, 1., norm_type=cv2.NORM_MINMAX).astype(np.float32)
    return mask

all_images = glob.glob(join(BASE_PATH, "*R.jpg"))

for fname in all_images:
    bname = basename(fname)
    orig_fname = bname.split(".")[0][:-1]
    mask_name = orig_fname + "M" + ".jpg"
    orig_fname += ".jpg"
    image = cv2.imread(join(BASE_PATH, orig_fname))
    mask = get_mask(image)
    cv2.imwrite(join(BASE_PATH, mask_name), mask)

    #image_resize = cv2.resize(image,(1024,1024))
    #no_hair = hair_remove(image_resize)
    #no_hair_512 = cv2.resize(no_hair, (512, 512))
    #new_name = bname.split(".")[0] + 'R' + ".jpg"
    #cv2.imwrite(join(BASE_PATH, new_name), no_hair_512)
