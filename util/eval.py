import numpy as np
import os
import torch
import cv2
import math
import scipy
from numpy.ma.core import exp
from scipy.constants.constants import pi
from skimage.metrics import structural_similarity as ssim
def rgb2xyz(rgb):
    def format(c):
        c = c / 255.
        if c > 0.04045: c = ((c + 0.055) / 1.055) ** 2.4
        else: c = c / 12.92
        return c * 100
    rgb = list(map(format, rgb))
    xyz = [None, None, None]
    xyz[0] = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    xyz[1] = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    xyz[2] = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505
    return xyz

# Converts XYZ pixel array to LAB format.
# Implementation derived from http://www.easyrgb.com/en/math.php
def xyz2lab(xyz):
    def format(c):
        if c > 0.008856: c = c ** (1. / 3.)
        else: c = (7.787 * c) + (16. / 116.)
        return c
    xyz[0] = xyz[0] / 95.047
    xyz[1] = xyz[1] / 100.00
    xyz[2] = xyz[2] / 108.883
    xyz = list(map(format, xyz))
    lab = [None, None, None]
    lab[0] = (116. * xyz[1]) - 16.
    lab[1] = 500. * (xyz[0] - xyz[1])
    lab[2] = 200. * (xyz[1] - xyz[2])
    return lab

# Converts RGB pixel array into LAB format.
def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

# Returns CIEDE2000 comparison results of two LAB formatted colors.
# Translated from CIEDE2000 implementation in https://github.com/markusn/color-diff
def ciede2000(lab1, lab2):
    def degrees(n): return n * (180. / np.pi)
    def radians(n): return n * (np.pi / 180.)
    def hpf(x, y):
        if x == 0 and y == 0: return 0
        else:
            tmphp = degrees(np.arctan2(x, y))
            if tmphp >= 0: return tmphp
            else: return tmphp + 360.
        return None
    def dhpf(c1, c2, h1p, h2p):
        if c1 * c2 == 0: return 0
        elif np.abs(h2p - h1p) <= 180: return h2p - h1p
        elif h2p - h1p > 180: return (h2p - h1p) - 360.
        elif h2p - h1p < 180: return (h2p - h1p) + 360.
        else: return None
    def ahpf(c1, c2, h1p, h2p):
        if c1 * c2 == 0: return h1p + h2p
        elif np.abs(h1p - h2p) <= 180: return (h1p + h2p) / 2.
        elif np.abs(h1p - h2p) > 180 and h1p + h2p < 360: return (h1p + h2p + 360.) / 2.
        elif np.abs(h1p - h2p) > 180 and h1p + h2p >= 360: return (h1p + h2p - 360.) / 2.
        return None
    L1 = lab1[0]
    A1 = lab1[1]
    B1 = lab1[2]
    L2 = lab2[0]
    A2 = lab2[1]
    B2 = lab2[2]
    kL = 1
    kC = 1
    kH = 1
    C1 = np.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = np.sqrt((A2 ** 2.) + (B2 ** 2.))
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - np.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25. ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = np.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = np.sqrt((a2P ** 2.) + (B2 ** 2.))
    h1P = hpf(B1, a1P)
    h2P = hpf(B2, a2P)
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf(C1, C2, h1P, h2P)
    dHP = 2. * np.sqrt(c1P * c2P) * np.sin(radians(dhP) / 2.)
    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf(C1, C2, h1P, h2P)
    T = 1. - 0.17 * np.cos(radians(aHP - 39)) + 0.24 * np.cos(radians(2. * aHP)) + 0.32 * np.cos(radians(3. * aHP + 6.)) - 0.2 * np.cos(radians(4. * aHP - 63.))
    dRO = 30. * np.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = np.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / np.sqrt(20. + ((aL - 50.) ** 2.)))
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * np.sin(radians(2. * dRO))
    return np.sqrt(((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) + ((dHP / (sH * kH)) ** 2.) + rT * (dCP / (sC * kC)) * (dHP / (sH * kH)))


def ciede(img1, img2):
    c=0
    shape=img1.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            lab1=img1[i,j,:].astype(float)
            lab2=img2[i,j,:].astype(float)
            c+=ciede2000(lab1,lab2)
    return c/shape[0]/shape[1]


def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def eval(gtpath,dehazedpath,use_ciede=True):
    lst1=os.listdir(gtpath)
    lst2=os.listdir(dehazedpath)
    p=0
    s=0
    n=45
    c=0
    for i in range(n):
        name1=str(i)+'.jpg'
        name2=str(i+1)
        if(name2.__len__()<2):
            name2='0'+name2
        name2=name2+'_outdoor_GT512.jpg'
        gt=cv2.imread(gtpath+name2)
        hazy=cv2.imread(dehazedpath+name1)
        print(name1)
        print(name2)
        p += psnr(gt,hazy)
        s += ssim(gt,hazy,multichannel=True)
        if(use_ciede==True):
            c+=ciede(gt,hazy)
    print('the average psnr is', p/n)
    print('the average ssim is', s/n)
    print('the average ciede2000 is', c/n)

eval("D:/dadehazing/DA_dahazing-master/DA_dahazing-master/datasets/other_test/o-haze/GT512/",
     "D:/dadehazing/DA_dahazing-master/DA_dahazing-master/results/run_test/ohaze/")
