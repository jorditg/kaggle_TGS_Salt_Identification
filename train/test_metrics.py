#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 08:48:25 2018

@author: jordi
"""

import loss
import numpy as np
import torch
from PIL import Image

def main():
    dir="./img_test_metrics/"

    i1=Image.open(dir + "circle_centered.jpeg")
    i2=Image.open(dir + "circle_half.jpeg")
    
    i1=np.array(i1.getdata()).astype("float32")/255.
    i2=np.array(i2.getdata()).astype("float32")/255.
    i1 = torch.from_numpy(i1).reshape(299,299,3).permute(2,0,1)
    i2 = torch.from_numpy(i2).reshape(299,299,3).permute(2,0,1)
    
    l = loss.ISICLoss()
    
    print(l.forward(i1.unsqueeze(0), i2.unsqueeze(0)))

if __name__ == "__main__":
    main()