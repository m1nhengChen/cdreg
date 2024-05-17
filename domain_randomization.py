import torch
import math
import random
import numpy as np
import torch.nn.functional as F
import torchvision
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
 
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.mean = 0
        self.sl = sl
        self.sh = sh
        self.r1 = r1
 
    def __call__(self, img):
 
        if random.uniform(0, 1) > self.probability:
            return img
        self.mean= img.mean()
        for attempt in range(100):
            area = img.size()[2] * img.size()[3]
 
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
 
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
 
            if w < img.size()[3] and h < img.size()[2]:
                x1 = random.randint(0, img.size()[2] - h)
                y1 = random.randint(0, img.size()[3] - w)
                # if img.size()[1] == 3:
                #     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                #     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                #     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                # else:
                img[:,:, x1:x1 + h, y1:y1 + w] = self.mean
                return img
 
        return img


def domain_randomization(I, device, invert=False):
    if invert:
        I=invert(I,p=0.5)
    I=smoothing(I,device=device,p=0.5)
    I=gaussian_noise(I,p=0.5)
    I=renormalization(I,p=0.5)
    I=linear_scaling(I,p=0.5)
    I=gamma_transform(I,p=0.5)
    I=nonlinear_scaling(I,p=0.5)
    return I


def invert(I,p=0.5):
    if random.uniform(0,1)>p:
        return I
    I_i = torch.max(I) - I
    return I_i


def gaussian_noise_injection(I, device):
    delta = random.uniform(0.005, 0.1) * torch.max(I)
    mean = torch.zeros_like(I, dtype=torch.float, device=device)
    std = torch.full_like(I, delta, dtype=torch.float, device=device)
    I_gni = I + torch.normal(mean=mean, std=std)
    return I_gni


def gamma_transform(I,p=0.5):
    if random.uniform(0,1)>p:
        return I
    I_n = (I - torch.min(I)) / (torch.max(I) - torch.min(I))
    gamma = random.uniform(0.7, 1.3)
    I_gt = I_n ** gamma
    I_gt = torch.clamp(I_gt, 0, 1) 
    return I_gt


def box_corruption(I):
    size = 32
    x_center = random.randint(0, 255)
    y_center = random.randint(0, 255)
    I_bc = I
    if x_center <= size / 2:
        if y_center <= size / 2:
            I_bc[:, :, :size, :size] = 1
        elif y_center >= 255 - size / 2:
            I_bc[:, :, :size, -size:] = 1
        else:
            I_bc[:, :, :size, y_center - size / 2:y_center + size / 2 + 1] = 1
    elif x_center >= 255 - size / 2:
        if y_center <= size / 2:
            I_bc[:, :, -size:, :size] = 1
        elif y_center >= 255 - size / 2:
            I_bc[:, :, -size:, -size:] = 1
        else:
            I_bc[:, :, -size:, y_center - size / 2:y_center + size / 2 + 1] = 1
    else:
        if y_center <= size / 2:
            I_bc[:, :, x_center - size / 2:x_center + size / 2 + 1, :size] = 1
        elif y_center >= 255 - size / 2:
            I_bc[:, :, x_center - size / 2:x_center + size / 2 + 1, -size:] = 1
        else:
            I_bc[:, :, x_center - size / 2:x_center + size / 2 + 1, y_center - size / 2:y_center + size / 2 + 1] = 1
    return I_bc


def smoothing(I, device,p=0.5, sigma = 1):
    if random.uniform(0,1)>p:
        return I
    kernel3x3_flag = random.choice((True, False))
    if kernel3x3_flag:
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - 1 * sigma) ** 2 + (y - 1 * sigma) ** 2) / (2 * sigma ** 2)), (3, 3))
        kernel = torch.tensor(kernel, dtype=torch.float, requires_grad=False, device=device)
        kernel = kernel.view(1, 1, 3, 3)
        I = F.conv2d(I, kernel, padding = 1)
        return torch.clamp(I, 0, 1)
    else:
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - 2 * sigma) ** 2 + (y - 2 * sigma) ** 2) / (2 * sigma ** 2)), (5, 5))
        kernel = torch.tensor(kernel, dtype=torch.float, requires_grad=False, device=device)
        kernel = kernel.view(1, 1, 5, 5)
        I = F.conv2d(I, kernel, padding = 2)
        return torch.clamp(I, 0, 1)

def linear_scaling(I, p=0.5):
    if random.uniform(0,1)>p:
        return I
    # rndomly sample the scale factor
    scale_factor = torch.rand(1) * 0.15 + 0.9 
    # linear scaling
    scaled_image = I * scale_factor
    scaled_image = torch.clamp(scaled_image, 0, 1) 
    return scaled_image

def renormalization(I,p=0.5):
    if random.uniform(0,1)>p:
        return I
    lower_bound = random.uniform(-0.04 * torch.max(I), 0.02 * torch.max(I))
    upper_bound = random.uniform(0.9 * torch.max(I), 1.05 * torch.max(I))
    I = (I - lower_bound) / (upper_bound - lower_bound)
    I = torch.clamp(I, 0, 1) 
    return I


def gaussian_noise(I,p=0.5):
    if random.uniform(0,1)>p:
        return I
    # Get the maximum value of an image tensor
    max_value = torch.max(I)

    # Sample value of mean uniformly from range
    min_mean = -0.15 * max_value
    max_mean = 0.1 * max_value
    mean = torch.empty(1).uniform_(min_mean.item(), max_mean.item())

    # Generate Gaussian noise
    gaussian_noise = torch.randn_like(I) * mean

    # Add Gaussian noise to the image
    noisy_image = I + gaussian_noise*0.4
    noisy_image = torch.clamp(noisy_image, 0, 1)  # Limit pixel values to the range [0, 1]
    return noisy_image


def nonlinear_scaling(I, p=0.5):
    if random.uniform(0,1)>p:
        return I
    # Randomly sample a and b uniformly from (0.8, 1.1)
    a = torch.empty(1).uniform_(0.8, 1.1)
    b = torch.empty(1).uniform_(0.8, 1.1)

    # Randomly sample c uniformly from (-0.5, 0.5)
    c = torch.empty(1).uniform_(-0.5, 0.4)
    # Calculate the non-linear scaling function a*sin(b*x + c)
    I= a * torch.sin(b * I + c)
    I = torch.clamp(I, 0, 1)  # Limit pixel values to the range [0, 1]
    return I
