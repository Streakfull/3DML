from pathlib import Path

import numpy as np
import k3d
from matplotlib import cm, colors
import PIL.Image
import IPython.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_occupancy(occupancy_grid, flip_axes=False):
    point_list = np.concatenate([c[:, np.newaxis]
                                for c in np.where(occupancy_grid)], axis=1)

    visualize_pointcloud(
        point_list, 1, flip_axes=flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_cloud, point_size, colors=None, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False,
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(
        np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    #import pdb;pdb.set_trace();
    plot.display()
    plt.show()


def visualize_image(image_array):
    plt.imshow(image_array)


def visualize_images(images, rows=5):
    nimages = images.shape[0]
    columns = nimages//rows
    if (nimages % rows != 0):
        rows += 1
    if(columns == 0):
        columns = 1
    fig = plt.figure(figsize=(20, 20))
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()
