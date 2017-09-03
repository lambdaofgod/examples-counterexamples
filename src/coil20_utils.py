"""
 this script assumes its functions are called from some of project's subdirectories
"""


import os
import load_data_utils
from skimage.io import imread_collection, concatenate_images
import matplotlib.pyplot as plt


def load_images():
    """
    Return COIL-20 images

    output:
     
    ndarray of shape (1440, 16384)
    
    (1440 flattened 128x128 images)
    """
    COIL20_SUFFIX = 'coil-20-proc'
    __, DATA_DIR = load_data_utils.get_env_vars()
    COIL20_DIR = DATA_DIR + '/' +  COIL20_SUFFIX
    
    image_list =  imread_collection(COIL20_DIR + '/*.png')
    images = concatenate_images(image_list)
    imshape = images.shape
    reshaped_images = images.reshape(imshape[0], imshape[1] * imshape[2])
    return reshaped_images 


def display_image(img, **kwds):
    """
    input:

    img 
    a 128x128 or 16384 ndarray
    """
    tmp_img = img.reshape(128, 128)
    plt.imshow(tmp_img, **kwds)
    plt.show()

