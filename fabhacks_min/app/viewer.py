from parts.cable import Cable
import polyscope as ps

def viewer():
    # set program options
    ps.set_program_name("macgyvering")
    ps.set_print_prefix("MACGYVER")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)
    ps.set_ground_plane_mode("none")
    # init and show the window
    ps.init()

    part = Cable()
    V, F = part.get_transformed_VF()
    ps.register_surface_mesh("Part{}-{}".format(id, part.__class__.__name__), V, F)
    for c in part.children:
        cV, cF = c.get_transformed_VF()
        ps.register_surface_mesh("Primitive{}-{}".format(c.id, c.__class__.__name__), cV, cF)
    ps.show()
