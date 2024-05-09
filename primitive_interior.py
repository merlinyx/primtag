import numpy as np
import torch

# note that all samplings in this file are interior samplings, which might make the chamfer loss to perform better
# WORK NOT KEEP UPDATED FURTHER...

def cylinder(parameters):
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

    random_coords = torch.rand(1000, 3)

    u = random_coords[:,0:1]
    v = random_coords[:,1:2]
    w = random_coords[:,2:3]
    v = v * length

    sampled_points = (center + (radius * w) * x_dir * torch.cos(2 * torch.pi * u) 
                             + (radius * w) * y_dir * torch.sin(2 * torch.pi * u) 
                             + axis * (v - length / 2) + axis / 2
    )

    return sampled_points


# Parameters: center (x,y,z), axis (a_x, a_y, a_z), inner radius, outer radius
# surface parameterization: u,v

# Equation:
# s(u,v) = C + radius * x_dir cos(u) + radius * y_dir sin(u) + axis * v
def torus(parameters):
    center = parameters[:3]
    axis = parameters[3:6]
    R = parameters[6]
    r = parameters[7]
    z_dir = axis / torch.linalg.norm(axis)

    if torch.linalg.norm(z_dir - torch.tensor([1., 0, 0])) < .000001:
        y_dir = torch.tensor([0, 1., 0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1., 0, 0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    random_coords = torch.rand(1000, 3)

    u = random_coords[:,0:1]
    v = random_coords[:,1:2]
    w = random_coords[:,2:3]

    # the return is in torch.Tensor, needed to convert to numpy() to do visualization
    return (center + 
              x_dir * (R * torch.cos(2 * torch.pi * u) + r * w * torch.cos(2 * torch.pi * u) * torch.cos(2 * torch.pi * v))
            + y_dir * (R * torch.sin(2 * torch.pi * u) + r * w * torch.sin(2 * torch.pi * u) * torch.cos(2 * torch.pi * v))
            + z_dir * r * w * torch.sin(2 * torch.pi * v)
            )

def rectangle(parameters):
    return rectangle_(parameters[:3], parameters[3:6], parameters[6], parameters[7], parameters[8])

# Center (x,y,z), axis (x,y,z), length, width, height
def rectangle_(center, axis, length, width, height, num_samples=1000):
    z_dir = axis / torch.norm(axis)
    if torch.allclose(z_dir, torch.tensor([1., 0, 0]), atol=1e-6):
        y_dir = torch.tensor([0, 1., 0])
    else:
        y_dir = torch.cross(z_dir, torch.tensor([1., 0, 0]), dim=-1)
    x_dir = torch.cross(y_dir, z_dir, dim=-1)

    random_coords = torch.rand(1000, 3)

    u = random_coords[:,0:1]
    v = random_coords[:,1:2]
    w = random_coords[:,2:3]

    return center + x_dir * length * (u - 0.5) + y_dir * width * (v - 0.5) + z_dir * height * (w - 0.5)


def hemisphere(parameters):
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

    random_coords = torch.rand(1000, 3)

    u = random_coords[:,0:1] * 2 * torch.pi
    v = random_coords[:,1:2] * torch.pi / 2
    w = random_coords[:,2:3] 

    # Calculate points on the hemisphere
    x = radius * w * torch.sin(v) * torch.cos(u)
    y = radius * w * torch.sin(v) * torch.sin(u)
    z = radius * w * torch.cos(v) 

    # Apply rotation based on the axis direction
    points = center + x * x_dir + y * y_dir + z * z_dir

    return points
