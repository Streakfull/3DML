"""Triangle Meshes to Point Clouds"""
import numpy as np






def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    A = vertices[faces[:,0]]
    B = vertices[faces[:,1]]
    C = vertices[faces[:,2]]
    areas = get_face_surface_area(A,B,C)
    areas_softmaxed = softmax(areas)
    face_indices = np.arange(0,faces.shape[0])
    sampled_face_indices = np.random.choice(face_indices, size=n_points,p=areas_softmaxed)
    sampled_faces = faces[sampled_face_indices]
    A = vertices[sampled_faces[:,0]]
    B = vertices[sampled_faces[:,1]]
    C = vertices[sampled_faces[:,2]]
    r1 = np.random.uniform(size=n_points).reshape((n_points,1))
    r2 = np.random.uniform(size=n_points).reshape((n_points,1))
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1-r2)
    w = np.sqrt(r1) * r2
    P = u*A + v*B + w*C
    return P


def get_face_surface_area(A,B,C):
    AB = B-A
    AC = C-A
    u = np.cross(AB,AC,axis=1)
    magnitude = np.linalg.norm(u,axis=1)
    area = magnitude/2
    return area


def softmax(x):
    num = np.exp(x - np.max(x))
    return num / num.sum()



# def sample_point_cloud(vertices, faces, n_points):
#     """
#     Sample n_points uniformly from the mesh represented by vertices and faces
#     :param vertices: Nx3 numpy array of mesh vertices
#     :param faces: Mx3 numpy array of mesh faces
#     :param n_points: number of points to be sampled
#     :return: sampled points, a numpy array of shape (n_points, 3)
#     """
#     areas = np.zeros(faces.shape[0])
#     points = np.zeros((n_points,3))
#     for i,face in enumerate(faces):
#         A = vertices[face[0]]
#         B = vertices[face[1]]
#         C = vertices[face[2]]
#         area = get_face_surface_area(A,B,C)
#         areas[i] = area
#     areas_softmaxed = softmax(areas)
#     face_indices = np.arange(0,faces.shape[0])
#     x = np.arange(faces.shape[0])
#     x  = softmax(x)
#     sampled_faces = np.random.choice(face_indices, size=n_points,p=areas_softmaxed)
#     for i in range(len(sampled_faces)):
#         sampled_triangle_index = sampled_faces[i]
#         sampled_triangle = faces[sampled_triangle_index]
#         A = vertices[sampled_triangle[0]]
#         B = vertices[sampled_triangle[1]]
#         C = vertices[sampled_triangle[2]]
#         r1 = np.random.uniform()
#         r2 = np.random.uniform()
#         u = 1 - np.sqrt(r1)
#         v = np.sqrt(r1)* (1-r2)
#         w = np.sqrt(r1)*r2
#         P = u*A + v*B + w*C
#         points[i] = P
#     return points