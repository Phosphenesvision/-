'''
@Description:  
@Author: Zhang Zhanpeng
@Github: https://github.com/phosphenesvision
@Date: 2020-07-09 22:21:37
@LastEditTime: 2020-07-14 11:56:51
'''
import os
import cv2
import math
import numpy as np

rootpath = '/home/phosphenesvision/FaceDetection/Pytorch_Retinaface/widerface_evaluate/ufdd_txt/'
savepath = '/home/phosphenesvision/FaceDetection/Pytorch_Retinaface/widerface_evaluate/ufdd_txt_changed/'

def singletxt(filename):
    txt_file = os.path.join(rootpath, filename)
    with open(txt_file, 'r') as file:
        
        a = file.readline()
        b = file.readline()
        lines = file.readlines()
        blines = []
        for l in lines:
            la = l.split(' ')
            la = la[:5]
            if float(la[4]) > 0.5:
                blines.append(l)
        count = len(blines)

    pred_file = os.path.join(savepath, filename)
    with open(pred_file, 'w') as file:
        file.write(a)
        file.write(str(count)+'\n')
        for l in blines:
            file.write(l)




if __name__ == '__main__':


    #rootpath = '/home/phosphenesvision/FaceDetection/UFDD/UFDD_val/images/illumination'
    #savepath = '/home/phosphenesvision/FaceDetection/UFDD/UFDD_val/changed/illumination'
    for dirname in os.listdir(rootpath):
        for filename in os.listdir(os.path.join(rootpath, dirname)):
            #print(filename)
            singletxt(os.path.join(dirname,filename))