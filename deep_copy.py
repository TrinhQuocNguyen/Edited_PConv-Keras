from copy import deepcopy
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

# Import modules from libs/ directory
from libs.util import random_mask_rectangles
from libs.util import str2bool

import os

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--size', default='512', type=int,
                    help='size of the image')
parser.add_argument('--folder_path', default='./data/road/GOPR0268_line_broken_02', type=str,
                    help='The folder path')
if __name__ == "__main__":
    
    args = parser.parse_args()
    print(args)

    # Load mask
    mask = cv2.imread(args.folder_path + "/mask/mask_mask.png")
    # process mask
    image_mask = deepcopy(mask)
    image_mask[image_mask <=128] = 128
    image_mask[image_mask > 128] = 0
    image_mask[image_mask > 0] = 255
    cv2.imwrite(args.folder_path + "/mask/" + "mask.jpg", image_mask)

    # prcocess images
    img_paths = os.listdir(args.folder_path + "/origin")
    for img_path in img_paths:
        # Load image
        img = cv2.imread(args.folder_path + "/origin/" + img_path)

        img = cv2.resize(img, (args.size,args.size))

        # Image + mask
        masked_img = deepcopy(img)
        masked_img[mask==0] = 255

        # cv2.imshow("image", mask*255)
        # cv2.imshow("img", img)

        # cv2.imshow("masked_img", masked_img)
        # cv2.waitKey(0)
        cv2.imwrite(args.folder_path + "/input/" + img_path, masked_img)





