import cv2
import numpy as np
import os
def concat(path1,savepath):
    lst = os.listdir(path1)
    for i in range(lst.__len__()):
        name=lst[i].split('_')
        name1=name[0]+'_outdoor_GT.jpg'
        name3=name[0]+'_outdoor_GT512.jpg'
        img1 = cv2.imread(path1+name1)
        img=cv2.resize(img1,(512,512))
        cv2.imwrite(savepath+name3,img)

if __name__=="__main__":
    concat("D:/dadehazing/DA_dahazing-master/DA_dahazing-master/datasets/other_test/o-haze/GT/",
           "D:/dadehazing/DA_dahazing-master/DA_dahazing-master/datasets/other_test/o-haze/GT512/")
