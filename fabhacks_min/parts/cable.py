from parts.part import Part, PartType, PartPrimitiveType
from assembly.frame import Frame
from primitives.rod import Rod
import os

class Cable(Part):
    def __init__(self):
        super().__init__()
        self.tid = PartType.Cable
        self.name_in_assembly = self.__class__.__name__ + "_" + str(self.id)
        self.part_obj_path = os.path.join(".", Part.temp_dir, "Cable.obj")
        self.collision_obj_path = self.part_obj_path
        self.rod1 = Rod({"length": 100, "radius": 1.2, "centerline": True}, parent_part=self)
        self.rod1.set_frame(Frame([0,0,-76], [0,0,0]))
        self.rod1.child_id = 0
        self.rod1.set_ohe(PartPrimitiveType.Rod1OfCable)
        self.rod2 = Rod({"length": 11, "radius": 2.5}, parent_part=self)
        self.rod2.set_frame(Frame([0,0,-20.5], [0,0,0]))
        self.rod2.child_id = 1
        self.rod2.set_ohe(PartPrimitiveType.Rod2OfCable)
        self.children = [self.rod1, self.rod2]
        self.children_varnames = ["rod1", "rod2"]
        self.children_pairwise_ranges = {(21, 22): (2.9883581585805428, 109.39141533936245)}
        self.primitive_types = ["Rod"]
        self.child_symmetry_groups = [0, 1]
        self.init_mesh_metrics()
