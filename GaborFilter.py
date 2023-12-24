import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import pprint
import os
from enum import Enum
import math as mth
from scipy import signal
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import itertools
from utils import *

# This implementation is based on the paper:"Defect detection in textured materials using Gabor filters" (Ajay Kumar, Grantham Pang). Defect detection in textured materials using gabor filters


FREQ_NUM = 4
THETAS_NUM = 4

geo_mean = lambda img1, img2: np.sqrt(np.multiply(img1, img2))
min_max_normalization = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))


class GaborFilter:

    def __init__(self, scale_num=FREQ_NUM, rotate_steps=THETAS_NUM):
        self.freq_num = scale_num
        self.thetas_num = rotate_steps
        self.thetas = [(i + 1) / self.thetas_num for i in range(self.thetas_num)]

    def _build_filters(self, w, h, sigma_x, sigma_y, psi):
        "Get set of filters for GABOR"
        filters = []
        for x in range(self.freq_num):
            f_var = 1 / 2 ** x
            for t in self.thetas:
                theta = t * np.pi
                kernel = self._get_gabor_kernel(w, h, sigma_x, sigma_y, theta, f_var, psi)
                kernel = 2.0 * kernel / kernel.sum()
                filters.append(kernel)
        return filters

    def _get_gabor_kernel(self, w, h, sigma_x, sigma_y, theta, fi, psi):
        "getting gabor kernel with those values"
        # Bounding box
        kernel_size_x = w
        kernel_size_y = h
        (y, x) = np.meshgrid(np.arange(0, kernel_size_y), np.arange(0, kernel_size_x))
        # Rotation 
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        # Calculate the gabor kernel according the formulae
        gb = np.exp(-1.0 * (x_theta ** 2.0 / sigma_x ** 2.0 + y_theta ** 2.0 / sigma_y ** 2.0)) * np.cos(
            2 * np.pi * fi * x_theta + psi)
        return gb

    def _extract_features(self, img):
        "A vector of 2n elements where n is the number of theta angles"
        "and 2 is the number of frequencies under consideration"
        kernels = self._build_filters(img.shape[0], img.shape[1], 0.5, 0.5, np.pi / 2)
        fft_filters = [np.fft.fft2(i) for i in kernels]
        img_fft = np.fft.fft2(img)
        a = img_fft * fft_filters
        s = [np.fft.ifft2(i) for i in a]
        filtered = [p.real for p in s]
        return filtered

    def extract_features(self, img, normalize=True):
        filtered_images = self._extract_features(img)
        if normalize:
            normalized_filters = []
            for i, f_img in enumerate(filtered_images):
                m, s = f_img.mean(), f_img.std()
                # f_img[np.abs(f_img- m) < np.abs(s)] = 0 # paper threshold that not works for the detecting defects.

                normalized_filters.append(min_max_normalization(f_img))
            return normalized_filters
        return filtered_images

    def _fuse_freqs_by_bernouli_rule(self, filtered_imgs_split_by_freq):

        filters_combinations = list(itertools.combinations(range(self.thetas_num), 2))
        Ts = []
        for freq_filters in filtered_imgs_split_by_freq:
            pairs_mul_sum = np.sum(freq_filters, axis=0)
            for p1, p2 in filters_combinations:
                pairs_mul_sum += np.multiply(freq_filters[p1], freq_filters[p2])

            Ts.append(min_max_normalization(pairs_mul_sum))
        return Ts

    def _remove_false_alarm(self, Ts):
        if len(Ts) <= 1:
            return Ts
        cleaned_Ts = []
        for i in range(3):
            cleaned_Ts.append(geo_mean(Ts[i], Ts[i + 1]))
        return cleaned_Ts

    def filter(self, img):
        normalized_filters = self.extract_features(img, normalize=True)

        freq_split = [normalized_filters[i * self.thetas_num:(i + 1) * self.thetas_num] for i in range(self.freq_num)]
        Ts = self._fuse_freqs_by_bernouli_rule(freq_split)  # step 1 in fusion. out with size = freq_num

        Ts = self._remove_false_alarm(Ts)  # step 2 in fusion. out with size = freq_num - 1
        Ts = (min_max_normalization(np.sum(Ts, axis=0) / 3) * 255).astype(
            "uint8")  # step 3 in fusion. out with size = 1
        return Ts
