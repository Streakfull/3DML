""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    # TODO: Your implementation ends here ###############
    mean_y = np.mean(pc_y,axis=0)
    mean_x = np.mean(pc_x,axis=0)
    centered_x = pc_x - mean_x
    centered_y = pc_y - mean_y
    t = mean_y
    N,_ = pc_x.shape
    X = centered_x.T
    Y = centered_y.T
   # print(centered_y.shape,centered_x.shape,"OK")
    cov = np.matmul(X,Y.T)
    u,d,v = np.linalg.svd(cov)
    det_u = np.linalg.det(u)
    det_v = np.linalg.det(v)
    product = det_u * det_v
    product = np.round(product,5)
    s = np.identity(3)
    if(product!=1):
        s[2,2] = -1
    print(u.shape,v.shape,"OKKK")
    
    #print(u,v,"OKK")
    R = np.dot(s,v)
    #R = np.dot(u,R)
    #R = np.dot(v.T,u.T)
    #R = np.dot(v.T,u.T)
    R = np.dot(R.T,u.T)
    t = mean_y - np.dot(R,mean_x)
   # return R,t

 
    print(u.shape,d.shape,v.shape,"SHAPE")
    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t
    pc_x_mean = np.mean(pc_x, axis=0)
    pc_y_mean = np.mean(pc_y, axis=0)
    pc_x_centered = pc_x - pc_x_mean
    pc_y_centered = pc_y - pc_y_mean
    # X = pc_x_centered.reshape((3,N))
    #Y = pc_y_centered.reshape((3,N))

    # Calculate the covariance matrix
    cov_matrix = pc_x_centered.T @ pc_y_centered
    # cov_matrix = X @ Y.T
    print(cov_matrix,"COV _MATRIX")
    # Singular value decomposition
    U, S, V_t = np.linalg.svd(cov_matrix)
    #print(U,V_t,"OKK??")
    # Rotation matrix
    #R = V_t.T @ U.T
    R = np.dot(V_t.T,U.T)
    # Translation vector
   # t = pc_y_mean - R @ pc_x_mean
    t = pc_y_mean - R @pc_x_mean

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
