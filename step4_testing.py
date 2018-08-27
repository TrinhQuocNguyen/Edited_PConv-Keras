
# coding: utf-8

# # Network Training
# Having implemented and tested all the components of the final networks in steps 1-3, 
# we are now ready to train the network on a large dataset (ImageNet).

# In[1]:


import gc
import datetime

import pandas as pd
import numpy as np

from copy import deepcopy
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from IPython.display import clear_output

from libs.pconv_model import PConvUnet
from libs.util import random_mask


plt.ioff()

# SETTINGS
TRAIN_DIR = "/home/ubuntu/trinh/Edited_Generative_Inpainting/training_data/training"
VAL_DIR = "/home/ubuntu/trinh/Edited_Generative_Inpainting/training_data/validation"
TEST_DIR = "/home/ubuntu/trinh/Edited_Generative_Inpainting/training_data/testing"


MASK_NOISE = "/home/ubuntu/trinh/Edited_Generative_Inpainting/test_dir/GOPR0076/mask/mask.jpg"

BATCH_SIZE = 4


# # Creating train & test data generator

# In[2]:


class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori

    def flow_from_directory_for_test(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            
            # Get augmentend image samples
            ori = next(generator)

            # Get masks for each image sample
            # mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)
            
            image_mask = cv2.imread(MASK_NOISE)
            image_mask = cv2.resize(image_mask,(512,512))
            image_mask[image_mask <=128] = 128
            image_mask[image_mask > 128] = 0
            image_mask[image_mask > 0] = 255
            
            
            mask = np.stack([image_mask for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori            
            
# Create training generator
train_datagen = DataGenerator(  
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip =True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)

# Create testing generator
test_datagen = DataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory_for_test(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, seed=1
)


# In[3]:


# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Show side by side
for i in range(len(ori)):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].imshow(masked[i,:,:,:])
    axes[1].imshow(mask[i,:,:,:])
    axes[2].imshow(ori[i,:,:,:])
    plt.show()

print("exit")
exit() 
# # Training on ImageNet

# In[4]:


def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""
    
    # Get samples & Display them        
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[2].imshow(ori[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')
                
        plt.savefig(r'data/test_samples/img_{}_{}.png'.format(i, pred_time))
        plt.close()


# ## Phase 1 - with batch normalization

# In[5]:


# Instantiate the model
model = PConvUnet(weight_filepath='data/logs/')
model.load("data/logs/85_weights_2018-08-04-11-30-41.h5")

# model.load(r"C:\Users\MAFG\Documents\Github-Public\PConv-Keras\data\logs\50_weights_2018-06-01-16-41-43.h5")


# In[8]:


# Run training for certain amount of epochs
model.fit(
    train_generator, 
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=50,        
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='data/logs/initial_training', write_graph=False)
    ]
)


# ## Phase 2 - without batch normalization

# RUN FROM HERE

# In[5]:


# Load weights from previous run
model = PConvUnet(weight_filepath='data/logs/')
model.load(
    "/home/ubuntu/trinh/PConv-Keras/data/logs/119_weights_2018-08-09-07-06-34.h5",
    train_bn=False,
    lr=0.00005
)


# In[6]:


# Run training for certain amount of epochs
model.fit(
    train_generator, 
    steps_per_epoch=10000,
    validation_data=val_generator,
    validation_steps=100,
    epochs=20,        
    workers=3,
    plot_callback=plot_callback,
    callbacks=[
        TensorBoard(log_dir='../data/logs/fine_tuning', write_graph=False)
    ]
)


# ## Phase 3 - Generating samples

# In[70]:


from keras.preprocessing import image

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(512, 512))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]


    
#     image_mask = cv2.imread(MASK_NOISE)
#     image_mask = cv2.resize(image_mask,(512,512))
#     image_mask[image_mask <=128] = 128
#     image_mask[image_mask > 128] = 0
#     image_mask[image_mask > 0] = 255
    
#     mask = np.stack([image_mask for _ in range(ori.shape[0])], axis=0)


    mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)
        
    # Apply masks to all image sample
    masked = deepcopy(img_tensor)
    masked[mask==0] = 1
    
    if show:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(mask[i,:,:,:])
        axes[2].imshow(img_tensor[i,:,:,:])
        plt.show()
        
#         plt.imshow(mask[0])
#         plt.axis('off')
#         plt.show()
        
#         plt.imshow(masked[0])                           
#         plt.show()
        
#         plt.imshow(img_tensor[0])                           
#         plt.show()
        

    
    return [masked, mask]


# In[68]:


# Load weights from previous run
model = PConvUnet(weight_filepath='data/logs/')
model.load(
    "/home/ubuntu/trinh/PConv-Keras/data/logs/46_weights_2018-07-18-10-44-29.h5",
    train_bn=False,
    lr=0.00005
)


# In[72]:



import os

TEST_DIR_RUN = "/home/ubuntu/trinh/Edited_Generative_Inpainting/training_data/testing/GOPR0037_taken"
dir_files = os.listdir(TEST_DIR_RUN)
dir_files.sort()
count = 0

for training_item in dir_files:
    base_file_name = os.path.basename(training_item) 
    print(TEST_DIR_RUN +"/" + base_file_name)
    img_path = TEST_DIR_RUN +"/" + base_file_name    # dog
    # load a single image
    new_image = load_image(img_path, True)

    # check prediction
    pred = model.predict(new_image)
    print(pred.shape)
    
    plt.figure(figsize=(4.5,4.5))
    plt.imshow(pred[0,:,:,:] * 1., interpolation='nearest')
    plt.show()
    count +=1
    if count> 5:
        break
    


# In[108]:


for i in range(100):
    print (i)
    img_path = '/home/ubuntu/trinh/Edited_Generative_Inpainting/test_dir/GOPR0076/input/raindrop0394.jpg'    # dog
    # load a single image
    new_image = load_image(img_path, True)

    # check prediction
    pred = model.predict(new_image)
    print(pred.shape)
    plt.imshow(pred[0,:,:,:] * 1., interpolation='nearest')
    plt.show()


# In[69]:


n = 0
for (masked, mask), ori in tqdm(test_generator):
    
    # Run predictions for this batch of images
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(pred_img.shape)
    plt.imshow(pred_img[0,:,:,:] * 1., interpolation='nearest')
    plt.show()
    
    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[0].xaxis.set_major_formatter(NullFormatter())
        axes[0].yaxis.set_major_formatter(NullFormatter())
        axes[1].xaxis.set_major_formatter(NullFormatter())
        axes[1].yaxis.set_major_formatter(NullFormatter())
                
#         plt.savefig(r'data/results/img_{}_{}.png'.format(i, pred_time))
        plt.savefig(r'data/results/img_{}_{}.png'.format(i, pred_time))
        plt.close()
        n += 1
        
    # Only create predictions for about 100 images
    if n > 2:
        break

