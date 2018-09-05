from random import randint
import itertools
import numpy as np
import cv2


def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.uint8)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(1, 10)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
    # Draw random rectangles
    for _ in range(randint(1, 10)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        # thickness = randint(3, size)
        thickness = -1
        if (-width/3 < (x1 - x2) < width/3) and -height/3 < (y1 - y2) < height/3:
            cv2.rectangle(img, (x1,y1), (x2,y2), (1,1,1), thickness)        
    
    return 1-img

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    nonblack = img.any(axis=-1).sum()

    percent = nonblack/ (img.shape[0]*img.shape[1]) * 100
    return percent

def random_mask_rectangles(height, width, channels=3, percent_from=10., percent_to=20., only_rec = True, short_rec= False):
    """Generate the random mask based on percentage of the entire image
    
    Arguments:
        height {int} -- height of the image
        width {int} -- width of the image
    
    Keyword Arguments:
        channels {int} -- chanel of the image (default: {3})
        percent_from {float} -- how many percent <from> (default: {10.})
        percent_to {float} -- how many percent <to> (default: {20.})
        only_rec {bool} -- only draw rectangles (default: {True})
    """        
    while True:
        img = np.zeros((height, width, channels), np.uint8)


        # Draw random rectangles
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, width), randint(1, width)
            y1, y2 = randint(1, height), randint(1, height)
            # thickness = randint(3, size)
            thickness = -1
            if short_rec == True:
                if (-width/3 < (x1 - x2) < width/3) and -height/3 < (y1 - y2) < height/3:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (1,1,1), thickness) 
            else:
                cv2.rectangle(img, (x1,y1), (x2,y2), (1,1,1), thickness)     
        
        if (only_rec != True):   
            # Set size scale
            size = int((width + height) * 0.03)
            if width < 64 or height < 64:
                raise Exception("Width and Height of mask must be at least 64!")
                
            # Draw random lines
            for _ in range(randint(1, 10)):
                x1, x2 = randint(1, width), randint(1, width)
                y1, y2 = randint(1, height), randint(1, height)
                thickness = randint(3, size)
                cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
                
            # Draw random circles
            for _ in range(randint(1, 10)):
                x1, y1 = randint(1, width), randint(1, height)
                radius = randint(3, size)
                cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
                
            # Draw random ellipses
            for _ in range(randint(1, 15)):
                x1, y1 = randint(1, width), randint(1, height)
                s1, s2 = randint(1, width), randint(1, height)
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = randint(3, size)
                cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)



        if (percent_from <= count_nonblack_np(img) < percent_to):
            print ("percent: ", count_nonblack_np(img))
            break
        
    return 1-img

def str2bool(v):
    """convert string to bool
    
    Arguments:
        v {string} -- string which describes the bool var
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
