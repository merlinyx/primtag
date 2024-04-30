import numpy as np
from scipy.spatial.transform import Rotation
from .helpers import xyz_matrix, mat_to_euler, normalize_angle

class Frame:

    def __init__(self, pos=[0,0,0], rot=[0,0,0]):
        self.pos = pos
        self.rot = normalize_angle(rot)

    def __str__(self):
        # return "FRAME RAW: {} {}\nFRAME MAT: {}\n".format(self.pos, self.rot, self.homogeneous_mat())
        return "FRAME RAW: {} {}\n".format(self.pos, self.rot)

    def get_program(self):
        return "Frame([{},{},{}], [{},{},{}])".format(self.pos[0], self.pos[1], self.pos[2], self.rot[0], self.rot[1], self.rot[2])

    def apply_transform(self, V):
        rotatedV = Rotation.from_euler('xyz', self.rot, degrees=True).apply(V)
        transformedV = rotatedV + self.pos
        return transformedV

    def homogeneous_mat(self):
        mat = np.zeros((4, 4))
        mat[:3, :3] = xyz_matrix(self.rot[0], self.rot[1], self.rot[2])
        mat[:3, 3] = self.pos
        mat[3, 3] = 1.
        return mat

    def axis(self, i):
        assert 0 <= i <= 2
        return np.array(self.homogeneous_mat()[:3, i])

    def transform_frame(self, other):
        self_mat = self.homogeneous_mat()
        other_mat = other.homogeneous_mat()
        new_frame_mat = self_mat @ other_mat
        new_pos = new_frame_mat[:3, 3]
        new_rot = mat_to_euler(new_frame_mat[:3, :3])
        return Frame(new_pos, new_rot)

    def set_from_mat(self, mat):
        self.pos = mat[:3, 3]
        self.rot = normalize_angle(mat_to_euler(mat[:3, :3]))

    def to_numpy_array(self):
        return np.array(self.to_list())

    def to_list(self):
        return list(self.pos) + list(self.rot)

    def radians_rot(self):
        return np.array(self.rot) * np.pi / 180.0

    def diff_with(self, other):
        self.rot = normalize_angle(self.rot)
        selff = self.to_numpy_array()
        other.rot = normalize_angle(other.rot)
        otherf = other.to_numpy_array()
        return selff - otherf

class AlignmentFrame(Frame):
    def __init__(self, pos=[0, 0, 0], rot=[0, 0, 0], axes=[[]]):
        super().__init__(pos, rot)
        self.axes = axes
        self.named_axes = {}
        if [0] in self.axes:
            self.named_axes[(0,)] = "X"
        if [1] in self.axes:
            self.named_axes[(1,)] = "Y"
        if [2] in self.axes:
            self.named_axes[(2,)] = "Z"
        if [1,2] in self.axes:
            self.named_axes[(1,2)] = "Y-Z"

    def __str__(self):
        return "POS: {} ROT: {}\n".format(self.pos, self.rot)

    def flip_axis(self, axes):
        for axis in axes:
            if self.rot[axis] == 0:
                self.rot[axis] = 180
            elif self.rot[axis] == 180:
                self.rot[axis] = 0
            else:
                self.rot[axis] = -self.rot[axis]
