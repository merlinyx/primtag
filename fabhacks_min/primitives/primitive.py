import os
from importlib_resources import files

class Primitive:
    # auxiliary id for __eq__
    pid = 0
    # temp folder for meshes
    temp_dir = "temp"
    # openscad command line executable path
    # openscad = os.getenv("OPENSCAD_EXEC")
    openscad = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"
    # openscad primitives modules file path
    print("Current Directory:", os.getcwd())

    # modules = str(files('fabhacks_min').joinpath('primitives','primitives.scad'))
    modules = 'fabhacks_min/primitives/primitives.scad'
    # whether to enable printing for openscad commands
    verbose = False
    # epsilon factor of extra dimension for primitives to be used for selection in the UI
    epsilon = 0.04

    def __init__(self, parent_part=None):
        self.id = Primitive.pid
        self.tid = None # type id
        Primitive.pid += 1

        self.parent = parent_part
        self.child_id = None # the id in its parent's children list
        self.ohe = None # one-hot encoding of the part-primitive type
        self.program = ""
        self.ctype_str = ""
        self.base_frame = None
        self.use_reduced = True

        self.V = None
        self.F = None
        self.is_concave = False
        self.single_conn = True
        self.connected = False

    def __eq__(self, other):
        if isinstance(other, Primitive):
            return self.id == other.id
        return False

    def __str__(self):
        # return "Primitive-{}(Pid: {}, Parent: Part-{}-id-{}, Child id: {})".format(self.__class__.__name__, self.id, self.parent.__class__.__name__, self.parent.id, self.child_id)
        return "Primitive{}-{}".format(self.id, self.__class__.__name__)

    def set_ohe(self, ohe):
        self.ohe = ohe

    # initialize shape parameters
    # params: a dict of parameters {name, value}
    #         lengths params in mm, angle params in radians
    def init_params(self, params):
        pass
    
    # initialize V and F
    def init_mesh(self):
        pass

    def set_frame(self, f):
        self.base_frame = f

    def frame(self):
        return self.base_frame

    def get_transformed_VF(self, reload=False):
        if reload or self.V is None or self.F is None:
            self.init_mesh()
        # make sure primitive mesh can be selected in UI
        return self.parent.frame.apply_transform(self.frame().apply_transform(self.V * 1.02)), self.F
    
    def get_global_frame(self):
        return self.parent.frame.transform_frame(self.frame())

    def generalized_radius(self):
        return 0

    def critical_dim(self):
        return None

    def connector_critical_dim(self):
        return None

    # returns !is_open for all the primitives that have it defined
    # True = we treat it as something that would stay within bounds
    # False = we treat it as something that would fall off under physics
    def has_bounds(self):
        if hasattr(self, "is_open"):
            return not self.is_open
        return False
