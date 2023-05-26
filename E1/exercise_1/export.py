"""Export to disk"""

import numpy as np

def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """
    print(f"Creating Obj file for {path} ...")
    # write vertices starting with "v "
    # write faces starting with "f "
    with open(path,"w") as file:
        for vertix in vertices:
            string = np.array2string(vertix, separator=' ', formatter={'float_kind':lambda x: "%.3f" % x}) \
           .strip('[]')
            string = f"v {string}"
            file.write(string)
            file.write('\n')
        
        for face in faces:
            string =  np.array2string(face, separator=' ', formatter={'float_kind':lambda x: "%.3f" % x}) \
           .strip('[]')
            string = f"f {string}"
            file.write(string)
            file.write('\n')


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    export_mesh_to_obj(path,pointcloud,[])