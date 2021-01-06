import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
	return window

def SSIM1(img1, img2):
	(_, channel, _, _) = img1.size()
	window_size = 11
	pad = int(window_size/11)
	window = create_window(window_size, channel).to(img1.device)
	mu1 = F.conv2d(img1, window, padding = pad, groups = channel)
	mu2 = F.conv2d(img2, window, padding = pad, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = pad, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = pad, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = pad, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def SSIM(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
	# Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
	if val_range is None:
		if torch.max(img1) > 128:
			max_val = 255
		else:
			max_val = 1

		if torch.min(img1) < -0.5:
			min_val = -1
		else:
			min_val = 0
		L = max_val - min_val
	else:
		L = val_range

	padd = 0
	(_, channel, height, width) = img1.size()
	if window is None:
		real_size = min(window_size, height, width)
		window = create_window(real_size, channel=channel).to(img1.device)

	mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
	mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

	C1 = (0.01 * L) ** 2
	C2 = (0.03 * L) ** 2

	v1 = 2.0 * sigma12 + C2
	v2 = sigma1_sq + sigma2_sq + C2
	cs = torch.mean(v1 / v2)  # contrast sensitivity

	ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

	if size_average:
		ret = ssim_map.mean()
	else:
		ret = ssim_map.mean(1).mean(1).mean(1)

	if full:
		return ret, cs
	return ret



def PSNR(img1, img2):
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def CIEDE2000(Lab_1, Lab_2):
	'''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
	C_25_7 = 6103515625  # 25**7

	L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
	L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
	C1 = math.sqrt(a1 ** 2 + b1 ** 2)
	C2 = math.sqrt(a2 ** 2 + b2 ** 2)
	C_ave = (C1 + C2) / 2
	G = 0.5 * (1 - math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7)))

	L1_, L2_ = L1, L2
	a1_, a2_ = (1 + G) * a1, (1 + G) * a2
	b1_, b2_ = b1, b2

	C1_ = math.sqrt(a1_ ** 2 + b1_ ** 2)
	C2_ = math.sqrt(a2_ ** 2 + b2_ ** 2)

	if b1_ == 0 and a1_ == 0:
		h1_ = 0
	elif a1_ >= 0:
		h1_ = math.atan2(b1_, a1_)
	else:
		h1_ = math.atan2(b1_, a1_) + 2 * math.pi

	if b2_ == 0 and a2_ == 0:
		h2_ = 0
	elif a2_ >= 0:
		h2_ = math.atan2(b2_, a2_)
	else:
		h2_ = math.atan2(b2_, a2_) + 2 * math.pi

	dL_ = L2_ - L1_
	dC_ = C2_ - C1_
	dh_ = h2_ - h1_
	if C1_ * C2_ == 0:
		dh_ = 0
	elif dh_ > math.pi:
		dh_ -= 2 * math.pi
	elif dh_ < -math.pi:
		dh_ += 2 * math.pi
	dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

	L_ave = (L1_ + L2_) / 2
	C_ave = (C1_ + C2_) / 2

	_dh = abs(h1_ - h2_)
	_sh = h1_ + h2_
	C1C2 = C1_ * C2_

	if _dh <= math.pi and C1C2 != 0:
		h_ave = (h1_ + h2_) / 2
	elif _dh > math.pi and _sh < 2 * math.pi and C1C2 != 0:
		h_ave = (h1_ + h2_) / 2 + math.pi
	elif _dh > math.pi and _sh >= 2 * math.pi and C1C2 != 0:
		h_ave = (h1_ + h2_) / 2 - math.pi
	else:
		h_ave = h1_ + h2_

	T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(
		3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

	h_ave_deg = h_ave * 180 / math.pi
	if h_ave_deg < 0:
		h_ave_deg += 360
	elif h_ave_deg > 360:
		h_ave_deg -= 360
	dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25) ** 2))

	R_C = 2 * math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7))
	S_C = 1 + 0.045 * C_ave
	S_H = 1 + 0.015 * C_ave * T

	Lm50s = (L_ave - 50) ** 2
	S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
	R_T = -math.sin(dTheta * math.pi / 90) * R_C

	k_L, k_C, k_H = 1, 1, 1

	f_L = dL_ / k_L / S_L
	f_C = dC_ / k_C / S_C
	f_H = dH_ / k_H / S_H

	dE_00 = math.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)
	return dE_00

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