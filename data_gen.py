
"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators.

No image processing operations are performed here, just load data from local directory
in batches.

"""

#from tifffile import imsave, imread
import os
import numpy as np


def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(img_dir + image_name)
            images.append(image.astype(np.float32))  # Ensure float32 type
    images = np.array(images)
    return images

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size



############################################

#Test the generator

from matplotlib import pyplot as plt
import random

train_img_dir = r"/home/d_rutvik/BraTS2020/input_data_2/train/images/" #output 128 folder: train folder
train_mask_dir = r"/home/d_rutvik/BraTS2020/input_data_2/train/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()


img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()