import numpy as np
import torch


# Parameters:
# center (x,y,z), axis (a_x, a_y, a_z), radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * x_dir cos(u) + radius * y_dir sin(u) + axis * v
def cylinder(parameters, coordinates):
    center = parameters[:3]
    axis = parameters[3:6]
    radius = parameters[6]
    length = parameters[7]
    z_dir = axis / torch.linalg.norm(axis)

    if torch.linalg.norm(z_dir - torch.tensor([1.,0,0])) < .000001:
        y_dir = torch.tensor([0,1.,0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1.,0,0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    u = coordinates[:,:1]
    v = coordinates[:,1:]

    v = v * length - length / 2

    return center + radius * x_dir * torch.cos(2*torch.pi*u) + radius * y_dir * torch.sin(2*torch.pi*u) + axis * v



# Parameters: center (x,y,z), axis (a_x, a_y, a_z), inner radius, outer radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * x_dir cos(u) + radius * y_dir sin(u) + axis * v
def torus(parameters, coordinates):
    center = parameters[:3]
    axis = parameters[3:6]
    # inner_radius = parameters[6]
    # outer_radius = parameters[7]
    R = parameters[6]
    r = parameters[7]
    z_dir = axis / torch.linalg.norm(axis)

    if torch.linalg.norm(z_dir - torch.tensor([1., 0, 0])) < .000001:
        y_dir = torch.tensor([0, 1., 0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1., 0, 0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    u = coordinates[:, :1]
    v = coordinates[:, 1:]

    # the return is in torch.Tensor, needed to convert to numpy() to do visualization
    return (center + x_dir * (
                R * torch.cos(2 * torch.pi * u) + r * torch.cos(2 * torch.pi * u) * torch.cos(2 * torch.pi * v))
            + y_dir * (R * torch.sin(2 * torch.pi * u) + r * torch.sin(2 * torch.pi * u) * torch.cos(2 * torch.pi * v))
            + z_dir * r * torch.sin(2 * torch.pi * v)
            )


# helper function for rectangle primitive (edge)
def uniform_sampling_on_rectangle(vertices, num_samples=1000):
    """
    Perform uniform sampling on a rectangle defined by four vertices.
    
    Args:
        vertices (torch.Tensor): Tensor representing the four vertices of the rectangle (A, B, C, D).
                                  Each row contains the coordinates (x, y, z) of a vertex.
        num_samples (int): Number of points to sample. Default is 1000.
        
    Returns:
        sampled_points (torch.Tensor): Sampled points within the rectangle.
    """
    # Generate random points within the rectangle
    u = torch.rand(num_samples)
    v = torch.rand(num_samples)
    # Compute points within the rectangle
    points_on_rectangle = (vertices[0] + u.unsqueeze(1) * (vertices[1] - vertices[0]) +
                           v.unsqueeze(1) * (vertices[3] - vertices[0]))
    return points_on_rectangle

# Center (x,y,z), axis (x,y,z), length, width, height
def rectangle(center, axis, length, width, height, num_samples=1000):
    """Generate points on a cuboid's surface, oriented along a given axis."""
    # Calculate perpendicular direction vectors
    z_dir = axis / torch.norm(axis)
    if torch.allclose(z_dir, torch.tensor([1., 0, 0]), atol=1e-6):
        y_dir = torch.tensor([0, 1., 0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1., 0, 0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    # Define the vertices of the cuboid
    half_sizes = torch.tensor([length, width, height]) / 2
    direction_vectors = torch.stack([x_dir, y_dir, z_dir]) * half_sizes.unsqueeze(0)
    corners = torch.tensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=torch.float32)
    vertices = center + corners @ direction_vectors.T

    # Define faces using vertices
    faces_indices = [
        [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 3, 2], [4, 5, 7, 6]
    ]
    faces = [vertices[indices] for indices in faces_indices]

    # Sample points on each face
    sampled_points_on_faces = [uniform_sampling_on_rectangle(face, num_samples) for face in faces]

    # Combine points on all faces
    sampled_points = torch.cat(sampled_points_on_faces, dim=0)

    return sampled_points

'''
# Example for rectangle:
# Parameters: center, axis, length, width, height
parameters = torch.tensor([0., 0, 0, 0, 0, 1, 5, 3, 2])  # Center (x,y,z), axis (x,y,z), length, width, height
coordinates = torch.rand((1000, 2))  # This is not used in the corrected function
points = primitive.rectangle(parameters[:3], parameters[3:6], parameters[6], parameters[7], parameters[8]).numpy()
mp.plot(points, shading={'point_size':.2})
'''


# Parameters:
# center (x,y,z), axis (a_x, a_y, a_z), radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * sin(v)cos(u) * x_dir  + radius * sin(v)sin(u) * y_dir + radius*cos(v) * z_dir
def hemisphere(parameters, coordinates):
    center = parameters[:3]
    axis = parameters[3:6]
    radius = parameters[6]
    z_dir = axis / torch.linalg.norm(axis)  # Normalize the axis to get the z-direction

    # Set up perpendicular directions
    if torch.linalg.norm(z_dir - torch.tensor([1., 0, 0])) < .000001:
        y_dir = torch.tensor([0, 1., 0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1., 0, 0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    # Spherical coordinates
    u = coordinates[:, :1] * 2 * torch.pi  # u varies from 0 to 2*pi
    v = coordinates[:, 1:] * torch.pi / 2  # v varies from 0 to pi/2

    # Calculate points on the hemisphere
    x = radius * torch.sin(v) * torch.cos(u)
    y = radius * torch.sin(v) * torch.sin(u)
    z = radius * torch.cos(v)

    # Apply rotation based on the axis direction
    points = center + x * x_dir + y * y_dir + z * z_dir

    return points


