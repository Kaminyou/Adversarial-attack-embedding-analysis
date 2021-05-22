import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from collections import OrderedDict

def single_image_process(image_path):
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224)])
    transform_totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img_tensor = torch.unsqueeze(transform_totensor(img), axis=0)
    return np.array(img), img_tensor

def rgb_image_to_tensor(rgb_array):
    transform_totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
    im = Image.fromarray(rgb_array)
    img_tensor = torch.unsqueeze(transform_totensor(im), axis=0)
    return img_tensor
    

def reverse_transform(data, std=torch.Tensor([0.229, 0.224, 0.225]), mean=torch.Tensor([0.485, 0.456, 0.406]), rgb_limit=True):
    data = data * std.view(3, 1, 1) + mean.view(3, 1, 1)
    data = data.squeeze().numpy().transpose(1,2,0)
    if rgb_limit == True:
        data *= 255
        data = np.clip(data, 0, 255)
        data = data.astype(np.uint8)
    return data

def np_array_save_img(array, img_name):
    im = Image.fromarray(array)
    im.save(img_name)

def diff_calculate(img, adv_img, augment = 1):
    return np.abs(img.astype(int) - adv_img.astype(int)) * augment

def int_to_torch_long(x):
    return torch.Tensor([x]).long()

def state_dict_key_transform(model_path):
    old_state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = "model." + key
        new_state_dict[new_key] = value
    return new_state_dict
