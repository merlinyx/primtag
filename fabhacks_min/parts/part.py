from fabhacks_min.assembly.frame import Frame
from fabhacks_min.parts.material import Material

import os
import igl
import numpy as np
import dill as pickle
from importlib_resources import files
from enum import Enum, auto

class PartType(Enum):
    Cable = auto()

    def __str__(self):
        return str(self.value)

class PartPrimitiveType(Enum):
    Rod1OfCable = auto()
    Rod2OfCable = auto()

    def __str__(self):
        return str(self.value)

part_to_pprim = {
    PartType.Cable: [PartPrimitiveType.Rod1OfCable, PartPrimitiveType.Rod2OfCable],
}

class Part:
    # auxiliary id for __eq__
    pid = 0
    # temp folder for meshes
    temp_dir = "temp"
    # number of part-primitive types
    npptype = len(PartPrimitiveType)

    def __init__(self):
        self.id = Part.pid
        self.tid = None # should be one of PartType enum, except for Environment Part
        Part.pid += 1

        self.children = []
        self.children_pairwise_ranges = {}
        self.children_varnames = []
        self.name_in_assembly = None
        self.primitive_init_program = None
        self.constructor_program = None
        self.primitive_types = []
        self.frame = Frame()
        self.V = None
        self.F = None
        self.bbox = None
        self.com_offset = None
        self.attach_length = 0
        self.is_concave = False
        self.special_collision_mesh = False
        self.is_fixed = False
        self.mass = 0
        self.mass_ratio = 1.
        self.material = Material(1.)

    def __eq__(self, other):
        if isinstance(other, Part):
            return self.id == other.id
        return False
    
    def __str__(self):
        return self.__class__.__name__ + "_" + str(self.id)

    def set_from(self, other):
        self.id = other.id
        self.tid = other.tid
        self.children = other.children
        self.children_varnames = other.children_varnames
        self.children_pairwise_ranges = other.children_pairwise_ranges
        # self.name_in_assembly = other.name_in_assembly
        self.primitive_init_program = other.primitive_init_program
        self.constructor_program = other.constructor_program
        self.primitive_types = other.primitive_types
        self.frame = other.frame
        self.V = other.V
        self.F = other.F
        self.bbox = other.bbox
        self.com_offset = other.com_offset
        self.is_concave = other.is_concave
        self.is_fixed = other.is_fixed
        self.part_obj_path = other.part_obj_path
        self.collision_obj_path = other.collision_obj_path
        self.mass = other.mass
        self.material = other.material
        for i in range(len(self.children)):
            self.children[i].parent = self
            setattr(self, self.children_varnames[i], self.children[i])

    def connectors(self):
        conns = []
        for child in self.children:
            if not child.connected:
                conns.append(child)
            elif not child.single_conn:
                conns.append(child)
        return conns

    def get_child_varname(self, child_id):
        return self.children_varnames[child_id]

    def compute_mass(self, V, F):
        V2 = np.ndarray(shape = (V.shape[0] + 1, 3), dtype=np.float32)
        V2[:-1, :] = V
        V2[-1, :] = [0, 0, 0]
        F2 = np.ndarray(shape = (F.shape[0], 4), dtype=np.int32)
        F2[:, :3] = F
        F2[:, 3] = V.shape[0]
        vol = igl.volume(V2, F2)
        totalV = np.abs(np.sum(vol))
        self.mass = self.material.get_mass(totalV)

    # http://melax.github.io/volint.html
    def compute_inertia(self):
        com = np.zeros(3)
        volume = 0.
        for i in range(self.F.shape[0]):
            tri = self.F[i, :]
            A = np.array([self.V[tri[0], :], self.V[tri[1], :], self.V[tri[2], :]])
            vol = np.linalg.det(A)
            com += vol * (self.V[tri[0], :] + self.V[tri[1], :] + self.V[tri[2], :])
            volume += vol
        com /= volume * 4.

        volume = 0.
        diag = np.zeros(3)
        offd = np.zeros(3)
        for i in range(self.F.shape[0]):
            tri = self.F[i, :]
            A = np.array([self.V[tri[0], :]-com, self.V[tri[1], :]-com, self.V[tri[2], :]-com])
            vol = np.linalg.det(A)
            normal = np.cross(self.V[tri[1], :] - self.V[tri[0], :], self.V[tri[2], :] - self.V[tri[0], :])
            com_ctr = (self.V[tri[0], :] + self.V[tri[1], :] + self.V[tri[2], :])/3. - com
            if np.dot(normal, com_ctr) < 0:
                volume -= vol
            else:
                volume += vol
            for j in range(3):
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3
                diag[j] += (A[0,j]*A[1,j] + A[1,j]*A[2,j] + A[2,j]*A[0,j] + \
                            A[0,j]*A[0,j] + A[1,j]*A[1,j] + A[2,j]*A[2,j]) * vol
                offd[j] += (A[0,j1]*A[1,j2] + A[1,j1]*A[2,j2] + A[2,j1]*A[0,j2] + \
                            A[0,j1]*A[2,j2] + A[1,j1]*A[0,j2] + A[2,j1]*A[1,j2] + \
                            A[0,j1]*A[0,j2]*2 + A[1,j1]*A[1,j2]*2 + A[2,j1]*A[2,j2]*2) * vol
        diag /= volume * (60. / 6.)
        offd /= volume * (120. / 6.)
        inertia_attrib = {"ixx":str(diag[1]+diag[2]), "ixy":str(-offd[2]), "ixz":str(-offd[1]), \
                          "iyy":str(diag[0]+diag[2]), "iyz":str(-offd[0]), "izz":str(diag[0]+diag[1])}
        return inertia_attrib

    def compute_com(self, V, F):
        com = np.zeros(3)
        volume = 0.
        for i in range(F.shape[0]):
            tri = F[i, :]
            A = np.array([V[tri[0], :], V[tri[1], :], V[tri[2], :]])
            vol = np.linalg.det(A)
            com += vol * (V[tri[0], :] + V[tri[1], :] + V[tri[2], :])
            volume += vol
        com /= volume * 4.
        self.com_offset = com

    def init_mesh_metrics(self, classname=None):
        if classname is None:
            classname = self.__class__.__name__.lower()
        # mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.off".format(classname)))
        mesh_path = "fabhacks_min/parts/meshes/" + classname + ".off"
        print(mesh_path)
        if not os.path.exists(mesh_path):
            # mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.obj".format(classname)))
            mesh_path = "fabhacks_min/parts/meshes/" + classname + ".obj"
            print(mesh_path, "What?")
        if not os.path.exists(mesh_path):
            # mesh_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.stl".format(classname)))
            mesh_path = "fabhacks_min/parts/meshes/" + classname + ".stl"
            print(mesh_path, "Huh?")
        if not os.path.exists(self.part_obj_path):
            print(mesh_path, self.part_obj_path, "Ahha!")
            V, F = igl.read_triangle_mesh(mesh_path)
            igl.write_obj(self.part_obj_path, V, F)
            

        # mesh_metrics_path = str(files('fabhacks').joinpath("parts", "meshes", "{}.metrics".format(classname)))
        mesh_metrics_path = "fabhacks_min/parts/meshes/" + classname + ".metrics"
        metrics = {}
        if os.path.exists(mesh_metrics_path):
            with open(mesh_metrics_path, 'rb') as f:
                metrics = pickle.load(f)
            self.com_offset = metrics["com_offset"]
            self.attach_length = metrics["attach_length"]
            self.mass = metrics["mass"]
            self.bbox = metrics["bbox"]
        else:
            V, F = igl.read_triangle_mesh(mesh_path)
            self.compute_com(V, F)
            metrics["com_offset"] = self.com_offset
            if self.attach_length == 0:
                self.attach_length = .5 * igl.bounding_box_diagonal(V)
            metrics["attach_length"] = self.attach_length
            self.compute_mass(V, F)
            metrics["mass"] = self.mass# * self.mass_ratio
            self.bbox, _ = igl.bounding_box(V) # returns BV (2^3 by 3 numpy array), BF
            metrics["bbox"] = self.bbox
            with open(mesh_metrics_path, 'wb') as f:
                pickle.dump(metrics, f)

        if not self.special_collision_mesh and self.is_concave:
            self.create_vhacd_mesh()

    def init_mesh(self):
        self.V, self.F = igl.read_triangle_mesh(self.part_obj_path)

    def create_vhacd_mesh(self):
        obj_acd_path = self.part_obj_path.replace(".obj", "_acd.obj")
        if not os.path.exists(obj_acd_path):
            p.vhacd(self.part_obj_path, obj_acd_path, "log.txt", pca=True, mode=1)
        self.collision_obj_path = obj_acd_path

    def get_transformed_VF(self, reload=False):
        if reload or self.V is None or self.F is None:
            self.init_mesh()
        return self.frame.apply_transform(self.V), self.F
