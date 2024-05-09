from primitives.primitive import Primitive
from solid import *
from solid import import_scad
import igl
import subprocess
import os

class Rod(Primitive):
    def __init__(self, params, parent_part=None):
        self.radius = None
        self.length = None
        self.is_open = True # whether the rod has open ends
        self.is_open_for_connect = True # whether the rod has an open end for connecting
        super().__init__(parent_part)
        self.tid = 0
        self.ctype_str = "Rod"
        if params is None:
            params = self.default_params()
        self.init_params(params)

    def critical_dim(self):
        return self.length

    def connector_critical_dim(self):
        return self.radius

    def generalized_radius(self):
        return self.radius

    def default_params(self):
        return {"radius": 0.5, "length": 10.}

    def init_params(self, params):
        self.radius = params["radius"]
        self.length = params["length"]
        if "is_open" in params:
            self.is_open = params["is_open"]
        if "is_open_for_connect" in params:
            self.is_open_for_connect = params["is_open_for_connect"]
        elif "is_open" in params:
            self.is_open_for_connect = params["is_open"]
        self.single_conn = False
        self.program = "{{\"length\": {}, \"radius\": {}}}".format(params["length"], params["radius"])

    # initialize V and F
    def init_mesh(self):
        scadfile = import_scad(Primitive.modules)
        b = scadfile.rod(self.radius * (1. + Primitive.epsilon), self.length)
        os.makedirs(Primitive.temp_dir, exist_ok=True)
        scadpath = os.path.join(".", Primitive.temp_dir, "rod.scad")
        offpath = os.path.join(".", Primitive.temp_dir, "rod_{}_{}.off".format(self.radius * (1. + Primitive.epsilon), self.length))
        scad_render_to_file(b, scadpath)
        run_args = ["sudo", Primitive.openscad, scadpath, "-o", offpath] if Primitive.verbose else [Primitive.openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)
