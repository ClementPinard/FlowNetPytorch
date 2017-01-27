from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)

class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        w, h = inputs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        inputs[0] = inputs[0].crop((x1, y1, x1 + tw, y1 + th))
        inputs[1] = inputs[1].crop((x1, y1, x1 + tw, y1 + th))
        target = target[y1 : y1 + th, x1 : x1 + tw]
        return inputs,target

class Scale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, inputs, target):
        w, h = inputs[0].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            ratio = ow/w
        else:
            oh = self.size
            ow = int(self.size * w / h)
            ratio = oh/h

        inputs[0] = inputs[0].resize((ow, oh), self.interpolation)
        inputs[1] = inputs[1].resize((ow, oh), self.interpolation)
        
        target = ndimage.interpolation.zoom(target,ratio)
        target*=ow/w
        return inputs, target[:oh,:ow]

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, inputs,target):
        if self.padding > 0:
            inputs[0] = ImageOps.expand(inputs[0], border=self.padding, fill=0)
            inputs[1] = ImageOps.expand(inputs[1], border=self.padding, fill=0)

        w, h = inputs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0].crop((x1, y1, x1 + tw, y1 + th))
        inputs[1] = inputs[1].crop((x1, y1, x1 + tw, y1 + th))
        return inputs,target[y1 : y1 + th,x1 : x1 + tw]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, target):
        if random.random() < 0.5:
            input[0] = input[0].transpose(Image.FLIP_LEFT_RIGHT)
            input[1] = input[1].transpose(Image.FLIP_LEFT_RIGHT)
            target = target.fliplr()
            target[:,:,0]*=-1
        return inputs,target

class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, target):
        if random.random() < 0.5:
            input[0] = input[0].transpose(Image.FLIP_UP_DOWN)
            input[1] = input[1].transpose(Image.FLIP_UP_DOWN)
            target = target.flipud()
            target[:,:,1]*=-1
        return inputs,target

class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    resample: Default: PIL.Image.BILINEAR
    expand: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    Careful when rotating more than 45 degrees, w and h will be inverted
    """
    def __init__(self, angle, resample=Image.BILINEAR, expand=False, diff_angle=0):
        self.angle = angle
        self.resample = resample
        self.expand = expand
        self.diff_angle = diff_angle
        assert(angle+diff_angle < 45)

    def __call__(self, inputs,target):
        applied_angle  = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle + diff/2
        angle2 = applied_angle - diff/2

        w, h = inputs[0].size

        def rotate_flow(i,j,k):
            if k==0:
                return (i-w/2)*(diff*math.pi/180)
            else:
                return (j-h/2)*(-diff*math.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map
        inputs[0] = inputs[0].rotate(angle1,resample=self.resample, expand=self.expand)
        inputs[1] = inputs[1].rotate(angle2,resample=self.resample, expand=self.expand)
        target = ndimage.interpolation.rotate(target,reshape=False)

        return inputs,target

class RandomCropRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    A crop is done to keep same image ratio, and no black pixels
    angle: max angle of the rotation cannot be more than 180 degrees
    resample: Default: PIL.Image.BILINEAR
    """
    def __init__(self, angle, size, diff_angle=0, resample=Image.BILINEAR):
        self.angle = angle
        self.resample = resample
        self.expand = True
        self.diff_angle = diff_angle
        self.size = size

    def __call__(self, inputs,target):
        applied_angle  = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle + diff/2
        angle2 = applied_angle - diff/2
        angle1_rad = angle1*np.pi/180
        angle2_rad = angle2*np.pi/180

        w, h = inputs[0].size

        def rotate_flow(i,j,k):
            return k*(i-w/2)*(diff*np.pi/180) + (k-1)*(j-h/2)*(-diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = inputs[0].rotate(angle1,resample=self.resample, expand=True)
        inputs[1] = inputs[1].rotate(angle2,resample=self.resample, expand=True)
        target = ndimage.interpolation.rotate(target,angle1,reshape=True)
        #flow vectors must be rotated too!
        target_=np.array(target, copy=True)
        target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] - np.sin(angle1_rad)*target[:,:,1]
        target[:,:,0] = np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target[:,:,1]

        #keep angle1 and angle2 within [0,pi/2] with a reflection at pi/2: -1rad is 1rad, 2rad is pi - 2 rad
        angle1_rad = np.pi/2 - np.abs(angle1_rad%np.pi - np.pi/2)
        angle2_rad = np.pi/2 - np.abs(angle2_rad%np.pi - np.pi/2)

        c1 = np.cos(angle1_rad)
        s1 = np.sin(angle1_rad)
        c2 = np.cos(angle2_rad)
        s2 = np.sin(angle2_rad)
        c_diag = h/np.sqrt(h*h+w*w)
        s_diag = w/np.sqrt(h*h+w*w)

        ratio = c_diag/max(c1*c_diag+s1*s_diag,c2*c_diag+s2*s_diag)

        crop = CenterCrop((int(h*ratio),int(w*ratio)))
        scale = Scale(self.size)
        inputs, target = crop(inputs, target)
        return scale(inputs,target)

class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation
        

    def __call__(self, inputs,target):
        w,h = inputs[0].size
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw==0 and th==0:
            return inputs, target
        #compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,-tw), min(w-tw,w), max(0,tw), min(w+tw,w)
        y1,y2,y3,y4 = max(0,-th), min(h-th,h), max(0,th), min(h+th,h)

        inputs[0] = inputs[0].crop((x1, y1, x2, y2))
        inputs[1] = inputs[1].crop((x3, y3, x4, y4))

        target= target[y1:y2,x1:x2]
        target[:,:,0]+= tw
        target[:,:,1]+= th

        return inputs, target
