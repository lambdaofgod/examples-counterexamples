import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


def images_scatterplot(image_vectors, img_coordinates, imsize=(128, 128), figsize=(18, 12)):
  plt.figure(figsize=(20, 14))
  ax = plt.gca()
  plt.axis('off')
    
  plt.scatter(img_coordinates[:, 0], img_coordinates[:, 1])
  
  artists = []

  images = image_vectors.reshape(-1, *imsize)
  for i, (img, img_coords) in enumerate(zip(images, img_coordinates)):
    ab = AnnotationBbox(OffsetImage(img, cmap='gray', zoom=0.25), (img_coords[0], img_coords[1]), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
