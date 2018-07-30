
# coding: utf-8

# # Mask Generation with OpenCV
# In the paper they generate irregular masks by using occlusion/dis-occlusion between two consecutive frames of videos, 
# as described in [this paper](https://lmb.informatik.uni-freiburg.de/Publications/2010/Bro10e/sundaram_eccv10.pdf). 
# 
# Instead we'll simply be using OpenCV to generate some irregular masks, 
# which will hopefully perform just as well. We've implemented this in the function `random_mask`, 
# which is located in the `util.py` file int he libs directory

# In[3]:


import itertools
import matplotlib
import matplotlib.pyplot as plt
from libs.util import random_mask
from libs.util import random_mask_rectangles
from libs.util import str2bool

from random import randint
import numpy as np
import cv2
import argparse


# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./data/masks', type=str,
                    help='The folder path')
parser.add_argument('--quantity', default='100', type=int,
                    help='quantity')
parser.add_argument('--size', default='512', type=int,
                    help='size of the image')
parser.add_argument('--percent_from', default='10', type=int,
                    help='percent_from')
parser.add_argument('--only_rec', default=False, type=str2bool,
                    help='only_rec')
parser.add_argument('--short_rec', default=True, type=str2bool,
                    help='short_rec')


if __name__ == "__main__":
    
    args = parser.parse_args()
    print(args)

    for i in range(args.quantity):
        # Generate images
        img = random_mask_rectangles(args.size, args.size, 3, args.percent_from, args.percent_from + 10, args.only_rec, args.short_rec) 
        

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # write masks
        cv2.imwrite(args.folder_path + '/mask_size_{}_from_{}_rec_{}_{}.png'.format(args.size, args.percent_from, args.only_rec, i), img*255)

