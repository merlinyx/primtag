from solid import *
import igl
import subprocess
import os

# openscad command line executable path
openscad = os.getenv("OPENSCAD_EXEC")

class PrimitiveMesh:

    # openscad primitives modules file path
    modules = os.path.abspath("primitives.scad")
    # output directory of the intermediate files for initializing the mesh
    output_dir = "."

    def __init__(self, primitive_type: str, parameters: dict):
        """
        primitive_type: string, the type of the primitive
        parameters: dict, the parameters of the primitive

        V and F stores the mesh vertices and faces (np.ndarray)
        """
        self.primitive_type = primitive_type.lower()
        self.parameters = parameters
        self.V = None
        self.F = None

    def init_mesh(self):
        # set up file paths
        scadfile = import_scad(PrimitiveMesh.modules)
        temp_dir = os.path.join(PrimitiveMesh.output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        scadpath = os.path.join(temp_dir, f"{self.primitive_type}.scad")
        parameters_string = "_".join([str(v) for v in self.parameters.values()])
        offpath = os.path.join(temp_dir, f"{self.primitive_type}_{parameters_string}.off")
        # construct openscad files with the type and the parameters
        b = None
        if self.primitive_type == "clip":
            width = self.parameters["width"]
            height = self.parameters["height"]
            thickness = self.parameters["thickness"]
            dist = self.parameters["dist"]
            open_gap = self.parameters["open_gap"]
            b = scadfile.clip(width, height, thickness, dist, open_gap)
        elif self.primitive_type == "edge":
            width = self.parameters["width"]
            length = self.parameters["length"]
            height = self.parameters["height"]
            b = scadfile.edge(width, length, height)
        elif self.primitive_type == "hemisphere":
            radius = self.parameters["radius"]
            b = scadfile.hemisphere(radius)
        elif self.primitive_type == "hole":
            arc_radius = self.parameters["arc_radius"]
            thickness = self.parameters["thickness"]
            b = scadfile.hook(arc_radius, 360, thickness)
        elif self.primitive_type == "hook":
            arc_radius = self.parameters["arc_radius"]
            arc_angle = self.parameters["arc_angle"]
            thickness = self.parameters["thickness"]
            b = scadfile.hook(arc_radius, arc_angle, thickness)
        elif self.primitive_type == "rod":
            radius = self.parameters["radius"]
            length = self.parameters["length"]
            b = scadfile.rod(radius, length)
        elif self.primitive_type == "surface":
            width = self.parameters["width"]
            length = self.parameters["length"]
            b = scadfile.surface(width, length, 0.1)
        elif self.primitive_type == "tube":
            inner_radius = self.parameters["inner_radius"]
            thickness = self.parameters["thickness"]
            length = self.parameters["length"]
            b = scadfile.tube(inner_radius, thickness, length)
        if b is None:
            raise ValueError(f"Unknown primitive type: {self.primitive_type}")
        # render the openscad file to .off file and read the mesh
        scad_render_to_file(b, scadpath)
        run_args = [openscad, scadpath, "-o", offpath, "-q"]
        subprocess.run(run_args)
        self.V, self.F = igl.read_triangle_mesh(offpath)

if __name__ == "__main__":
    primitive_mesh = PrimitiveMesh("clip", {"width": 10, "height": 10, "thickness": 1, "dist": 1, "open_gap": 1})
    # primitive_mesh = PrimitiveMesh("edge", {"width": 10, "length": 10, "height": 10})
    # primitive_mesh = PrimitiveMesh("hemisphere", {"radius": 10})
    # primitive_mesh = PrimitiveMesh("hole", {"arc_radius": 10, "thickness": 1})
    # primitive_mesh = PrimitiveMesh("hook", {"arc_radius": 10, "arc_angle": 300, "thickness": 1})
    # primitive_mesh = PrimitiveMesh("rod", {"radius": 10, "length": 10})
    # primitive_mesh = PrimitiveMesh("surface", {"width": 10, "length": 10})
    # primitive_mesh = PrimitiveMesh("tube", {"inner_radius": 3, "thickness": 1, "length": 20})

    primitive_mesh.init_mesh()
    print(primitive_mesh.V)
    print(primitive_mesh.F)
    print(type(primitive_mesh.V))
    print(type(primitive_mesh.F))
