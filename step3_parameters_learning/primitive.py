import numpy as np
import torch


def clip(parameters, coordinates):
    center = parameters[:3]
    orientation = parameters[3:6]
    width = torch.abs(parameters[6])
    height = torch.abs(parameters[7])
    thickness = torch.abs(parameters[8])
    dist = torch.abs(parameters[9])
    open_gap = torch.abs(parameters[10])

    z_dir = orientation / torch.linalg.norm(orientation)
    x_dir = torch.tensor([1., 0, 0], dtype=torch.float32)
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=1e-6):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)
    x_dir = x_dir / torch.linalg.norm(x_dir)  # Ensure x_dir is normalized
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    y_dir = y_dir / torch.linalg.norm(y_dir)

    num_points = coordinates.shape[0]
    quarter_points = num_points // 4

    coords_face1 = coordinates[:quarter_points]
    coords_face2 = coordinates[quarter_points:2*quarter_points]
    coords_face3 = coordinates[2*quarter_points:3*quarter_points]
    coords_face4 = coordinates[3*quarter_points:]

    u1 = coords_face1[:, 0]
    v1 = coords_face1[:, 1]
    u2 = coords_face2[:, 0]
    v2 = coords_face2[:, 1]
    u3 = coords_face3[:, 0]
    v3 = coords_face3[:, 1]
    u4 = coords_face4[:, 0]
    v4 = coords_face4[:, 1]

    x1 = (u1 - 0.5) * width
    y1 = (v1 - 0.5) * height
    z1 = torch.zeros_like(x1) - dist / 2

    x2 = (u2 - 0.5) * width
    y2 = (v2 - 0.5) * height
    z2 = torch.zeros_like(x2) + dist / 2

    x3 = (u3 - 0.5) * width
    y3 = (v3 - 0.5) * height
    z3 = z1 + thickness

    x4 = (u4 - 0.5) * width
    y4 = (v4 - 0.5) * height
    z4 = z2 - thickness

    angle = open_gap / 2.0
    rotation_matrix1 = torch.tensor([
        [torch.cos(angle), -torch.sin(angle)],
        [torch.sin(angle), torch.cos(angle)]
    ], dtype=torch.float32)

    rotation_matrix2 = torch.tensor([
        [torch.cos(-angle), -torch.sin(-angle)],
        [torch.sin(-angle), torch.cos(-angle)]
    ], dtype=torch.float32)

    x1y1 = torch.stack([x1, y1], dim=-1)
    x2y2 = torch.stack([x2, y2], dim=-1)
    x3y3 = torch.stack([x3, y3], dim=-1)
    x4y4 = torch.stack([x4, y4], dim=-1)

    x1y1_rot = torch.matmul(x1y1, rotation_matrix1.T)
    x2y2_rot = torch.matmul(x2y2, rotation_matrix2.T)
    x3y3_rot = torch.matmul(x3y3, rotation_matrix1.T)
    x4y4_rot = torch.matmul(x4y4, rotation_matrix2.T)

    x1_rot, y1_rot = x1y1_rot[:, 0], x1y1_rot[:, 1]
    x2_rot, y2_rot = x2y2_rot[:, 0], x2y2_rot[:, 1]
    x3_rot, y3_rot = x3y3_rot[:, 0], x3y3_rot[:, 1]
    x4_rot, y4_rot = x4y4_rot[:, 0], x4y4_rot[:, 1]

    points_face1 = center.unsqueeze(0) + x1_rot.unsqueeze(-1) * x_dir.unsqueeze(0) + y1_rot.unsqueeze(-1) * y_dir.unsqueeze(0) + z1.unsqueeze(-1) * z_dir.unsqueeze(0)
    points_face2 = center.unsqueeze(0) + x2_rot.unsqueeze(-1) * x_dir.unsqueeze(0) + y2_rot.unsqueeze(-1) * y_dir.unsqueeze(0) + z2.unsqueeze(-1) * z_dir.unsqueeze(0)
    points_face3 = center.unsqueeze(0) + x3_rot.unsqueeze(-1) * x_dir.unsqueeze(0) + y3_rot.unsqueeze(-1) * y_dir.unsqueeze(0) + z3.unsqueeze(-1) * z_dir.unsqueeze(0)
    points_face4 = center.unsqueeze(0) + x4_rot.unsqueeze(-1) * x_dir.unsqueeze(0) + y4_rot.unsqueeze(-1) * y_dir.unsqueeze(0) + z4.unsqueeze(-1) * z_dir.unsqueeze(0)

    points = torch.cat([points_face1, points_face2, points_face3, points_face4], dim=0)
    return points


def tube(parameters, coordinates):
    center = parameters[:3]
    orientation = parameters[3:6]
    inner_radius = torch.abs(parameters[6])
    thickness = torch.abs(parameters[7])
    length = torch.abs(parameters[8])

    z_dir = orientation / torch.linalg.norm(orientation)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()

    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    y_dir = y_dir / torch.linalg.norm(y_dir)

    num_points = coordinates.shape[0]
    half_points = num_points // 2

    # Separate coordinates for inner and outer surfaces
    coordinates_inner = coordinates[:half_points]
    coordinates_outer = coordinates[half_points:]

    u_inner = coordinates_inner[:, 0:1]
    v_inner = coordinates_inner[:, 1:]
    u_outer = coordinates_outer[:, 0:1]
    v_outer = coordinates_outer[:, 1:]

    # inner surface of the tube
    theta_inner = 2 * torch.pi * u_inner
    r_inner = inner_radius
    x_inner = r_inner * torch.cos(theta_inner)
    y_inner = r_inner * torch.sin(theta_inner)
    z_inner = (v_inner - 0.5) * length

    # outer surface of the tube
    theta_outer = 2 * torch.pi * u_outer 
    r_outer = inner_radius + thickness
    x_outer = r_outer * torch.cos(theta_outer)
    y_outer = r_outer * torch.sin(theta_outer)
    z_outer = (v_outer - 0.5) * length

    points_inner = center.unsqueeze(0) + x_inner * x_dir.unsqueeze(0) + y_inner * y_dir.unsqueeze(0) + z_inner * z_dir.unsqueeze(0)
    points_outer = center.unsqueeze(0) + x_outer * x_dir.unsqueeze(0) + y_outer * y_dir.unsqueeze(0) + z_outer * z_dir.unsqueeze(0)

    points = torch.cat([points_inner, points_outer], dim=0)

    return points


def hook(parameters, coordinates):
    center = parameters[:3]
    orientation = parameters[3:6]
    arc_radius = torch.abs(parameters[6])
    angle = torch.abs(parameters[7])
    thickness = torch.abs(parameters[8])

    z_dir = orientation / torch.linalg.norm(orientation)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    # Generate valid x and y directions perpendicular to z_dir
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)  # Ensure x_dir is normalized
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    y_dir = y_dir / torch.linalg.norm(y_dir)

    u = coordinates[:, 0:1]
    v = coordinates[:, 1:]

    # Parametric equations for the hook (segment of a torus)
    theta = angle * u  # Angle for the hook segment
    phi = 2 * torch.pi * v

    R = arc_radius
    r = thickness

    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    points = center.unsqueeze(0) + x * x_dir.unsqueeze(0) + y * y_dir.unsqueeze(0) + z * z_dir.unsqueeze(0)
    return points




# Parameters: center (x,y,z), axis (a_x, a_y, a_z), inner radius, outer radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * x_dir cos(u) + radius * y_dir sin(u) + axis * v
def torus(parameters, coordinates):
    center = parameters[:3]
    orientation = parameters[3:6]
    arc_radius = torch.abs(parameters[6])
    thickness = torch.abs(parameters[7])

    z_dir = orientation / torch.linalg.norm(orientation)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    # Generate valid x and y directions perpendicular to z_dir
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    y_dir = y_dir / torch.linalg.norm(y_dir)

    u = coordinates[:, 0:1]
    v = coordinates[:, 1:]

    # Parametric equations for the torus
    theta = 2 * torch.pi * u
    phi = 2 * torch.pi * v

    R = arc_radius
    r = thickness

    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    points = center.unsqueeze(0) + x * x_dir.unsqueeze(0) + y * y_dir.unsqueeze(0) + z * z_dir.unsqueeze(0)
    return points



def cuboid(parameters, coordinates):
    center = parameters[:3]
    orientation = parameters[3:6]
    width = torch.abs(parameters[6])
    length = torch.abs(parameters[7])
    height = torch.abs(parameters[8])

    z_dir = orientation / torch.linalg.norm(orientation)
    # parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    # Generate valid x and y directions perpendicular to z_dir
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    y_dir = y_dir / torch.linalg.norm(y_dir)

    # Separate the coordinates for the six faces of the cuboid
    num_points = coordinates.shape[0]
    num_points_per_face = num_points // 6

    coords_face = lambda offset: coordinates[offset:offset+num_points_per_face]

    # Coordinates for each face
    coords_front = coords_face(0)
    coords_back = coords_face(num_points_per_face)
    coords_left = coords_face(2*num_points_per_face)
    coords_right = coords_face(3*num_points_per_face)
    coords_top = coords_face(4*num_points_per_face)
    coords_bottom = coords_face(5*num_points_per_face)

    def generate_face_points(coords, dir1, dir2, offset, dim1_scale, dim2_scale):
        u = coords[:, 0:1] - 0.5
        v = coords[:, 1:] - 0.5
        return center.unsqueeze(0) + u * dim1_scale * dir1.unsqueeze(0) + \
               v * dim2_scale * dir2.unsqueeze(0) + offset

    half_length = length / 2
    half_width = width / 2
    half_height = height / 2

    # Front and back faces
    points_front = generate_face_points(coords_front, x_dir, y_dir, half_height * z_dir.unsqueeze(0), length, width)
    points_back = generate_face_points(coords_back, x_dir, y_dir, -half_height * z_dir.unsqueeze(0), length, width)

    # Left and right faces
    points_left = generate_face_points(coords_left, z_dir, y_dir, -half_length * x_dir.unsqueeze(0), height, width)
    points_right = generate_face_points(coords_right, z_dir, y_dir, half_length * x_dir.unsqueeze(0), height, width)

    # Top and bottom faces
    points_top = generate_face_points(coords_top, x_dir, z_dir, half_width * y_dir.unsqueeze(0), length, height)
    points_bottom = generate_face_points(coords_bottom, x_dir, z_dir, -half_width * y_dir.unsqueeze(0), length, height)

    # Concatenate all face points
    points = torch.cat([points_front, points_back, points_left, points_right, points_top, points_bottom], dim=0)

    return points





# Parameters:
# center (x,y,z), axis (a_x, a_y, a_z), radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * sin(v)cos(u) * x_dir  + radius * sin(v)sin(u) * y_dir + radius*cos(v) * z_dir
def hemisphere_withBase(parameters, coordinates):  # work perfectly
    center = parameters[:3]
    axis = parameters[3:6]
    radius = torch.abs(parameters[6])

    z_dir = axis / torch.linalg.norm(axis)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir) 
    y_dir = torch.cross(z_dir, x_dir, dim=-1)

    # Separate coordinates for the hemisphere surface and the bottom face
    num_points = coordinates.shape[0]
    num_hemisphere_points = num_points // 2
    num_base_points = num_points - num_hemisphere_points

    u_surface = coordinates[:num_hemisphere_points, 0:1]
    v_surface = coordinates[:num_hemisphere_points, 1:]
    u_base = coordinates[num_hemisphere_points:, 0:1]
    v_base = coordinates[num_hemisphere_points:, 1:]

    # Generate points on the hemisphere surface
    theta_surface = torch.acos(1 - u_surface)  # Angle for hemisphere from 0 to pi/2
    phi_surface = 2 * torch.pi * v_surface

    x_surface = radius * torch.sin(theta_surface) * torch.cos(phi_surface)
    y_surface = radius * torch.sin(theta_surface) * torch.sin(phi_surface)
    z_surface = radius * torch.cos(theta_surface)

    points_surface = center.unsqueeze(0) + x_surface * x_dir.unsqueeze(0) + y_surface * y_dir.unsqueeze(0) + z_surface * z_dir.unsqueeze(0)

    # Generate points on the bottom face
    r_base = radius * torch.sqrt(u_base)
    phi_base = 2 * torch.pi * v_base

    x_base = r_base * torch.cos(phi_base)
    y_base = r_base * torch.sin(phi_base)
    z_base = torch.zeros_like(r_base)

    points_base = center.unsqueeze(0) + x_base * x_dir.unsqueeze(0) + y_base * y_dir.unsqueeze(0) + z_base * z_dir.unsqueeze(0)

    # Concatenate surface and base points
    points = torch.cat([points_surface, points_base], dim=0)

    return points

def hemisphere_NoBase(parameters, coordinates): # work perfectly
    center = parameters[:3]
    axis = parameters[3:6]
    radius = torch.abs(parameters[6])

    z_dir = axis / torch.linalg.norm(axis)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    # Generate a valid x direction perpendicular to z_dir
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)  # Ensure x_dir is normalized
    y_dir = torch.cross(z_dir, x_dir, dim=-1)

    u = coordinates[:, 0:1]
    v = coordinates[:, 1:]

    theta = torch.acos(1 - u)  # Angle for hemisphere from 0 to pi/2
    phi = 2 * torch.pi * v

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    points = center.unsqueeze(0) + x * x_dir.unsqueeze(0) + y * y_dir.unsqueeze(0) + z * z_dir.unsqueeze(0)
    return points


def cylinder(parameters, coordinates):
    center = parameters[:3]
    axis = parameters[3:6]
    radius = torch.abs(parameters[6])
    length = torch.abs(parameters[7])

    z_dir = axis / torch.linalg.norm(axis)
    parameters.detach().numpy()[3:6] = z_dir.detach().numpy()
    
    # Generate a valid x direction perpendicular to z_dir
    if torch.allclose(z_dir, torch.tensor([1., 0, 0], dtype=torch.float32), atol=0.000001):
        x_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        x_dir = torch.cross(z_dir, torch.tensor([0., 1, 0], dtype=torch.float32), dim=-1)
        if torch.linalg.norm(x_dir) < 1e-5:
            x_dir = torch.cross(z_dir, torch.tensor([0., 0, 1], dtype=torch.float32), dim=-1)

    x_dir = x_dir / torch.linalg.norm(x_dir)  # Ensure x_dir is normalized
    y_dir = torch.cross(z_dir, x_dir, dim=-1) 
    
    u = coordinates[:, 0:1]
    v = coordinates[:, 1:]
    v = v * length - length / 2

    points = center.unsqueeze(0) + radius * (torch.cos(2 * torch.pi * u) * x_dir.unsqueeze(0) +
                                             torch.sin(2 * torch.pi * u) * y_dir.unsqueeze(0)) + v * z_dir.unsqueeze(0)
    return points