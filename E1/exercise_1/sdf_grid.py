"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """
    boundary  = int(np.floor(resolution/2))
    index_boundary = boundary - 1
    axis = np.arange(-boundary+1,boundary+1,dtype=int)
    axis_index = np.linspace(-0.5, 0.5, resolution)
    shape = [resolution,resolution,resolution]
    grid = np.zeros(shape)
    for i in axis:
        for j in axis:
            for k in axis:
                value = sdf_function(np.array([axis_index[i+index_boundary]]),
                                     np.array([axis_index[j+index_boundary]]), 
                                     np.array([axis_index[k+index_boundary]]))
             
                grid[i+index_boundary][j+index_boundary][k+index_boundary] = value
    
    return grid
    #return grid

    # x = np.linspace(-0.5, 0.5, resolution)
    # y = np.linspace(-0.5, 0.5, resolution)
    # z = np.linspace(-0.5, 0.5, resolution)
    
    # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    # distances = sdf_function(xx, yy, zz)
    # return distances
    # ###############
    # TODO: Implement
    raise NotImplementedError
    # ###############
