import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread_collection, concatenate_images

from src import load_data_utils

COLORS = ['xkcd:neon green',
          'lightcoral',
          'xkcd:turtle green',
          'xkcd:cobalt',
          'xkcd:dusky pink',
          'xkcd:lighter purple',
          'xkcd:light cyan',
          'xkcd:khaki',
          'xkcd:green yellow',
          'xkcd:purple gray',
          'xkcd:very light blue',
          'xkcd:buff',
          'xkcd:mushroom']


def load_images():
    """
    Return COIL-20 images

    output:
     
    ndarray of shape (1440, 16384)
    
    (1440 flattened 128x128 images)
    """
    COIL20_SUFFIX = 'coil-20-proc'
    __, DATA_DIR = load_data_utils.get_env_vars(go_up=True)
    COIL20_DIR = DATA_DIR + '/' + COIL20_SUFFIX

    image_list = imread_collection(COIL20_DIR + '/*.png')
    images = concatenate_images(image_list)
    imshape = images.shape
    reshaped_images = images.reshape(imshape[0], imshape[1] * imshape[2])
    return reshaped_images


# plotting utils


def display_image(img, **kwds):
    """
    input:

    img 
    a 128x128 or 16384 ndarray
    """
    tmp_img = img.reshape(128, 128)
    plt.imshow(tmp_img, **kwds)
    plt.axis('off')
    plt.show()


def signed_scatterplot(images, labels, label_names):
    def plts(images):
        for c in np.unique(labels):
            idxs = labels == c
            plt.axis('off')
            yield plt.scatter(images[idxs, 0], images[idxs, 1], c=COLORS[c])

    plt.legend(plts(images), label_names)


def rotation_trajectories_scatterplot(images, labels, label_names):
    signed_scatterplot(images, labels, label_names)

    for l in np.unique(labels):
        idxs = labels == l
        l_imgs = images[idxs, :]
        for v1, v2 in zip(list(l_imgs[:-1]), list(l_imgs[1:])):
            plt.plot([v1[0], v2[0]], [v1[1], v2[1]], c=COLORS[l])
            plt.axis('off')
            v_first, v_last = l_imgs[0], l_imgs[-1]
        plt.plot([v_first[0], v_last[0]], [v_first[1], v_last[1]], c=COLORS[l])
        plt.axis('off')
