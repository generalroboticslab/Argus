import trimesh
import numpy as np
from odio_urdf import *
import os
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R
####################

def vector_to_rpy(direction):
    """
    Converts a direction vector to roll, pitch, and yaw angles.

    Args:
      direction: A numpy array of shape (3,) or (N, 3) representing the direction vector(s).

    Returns:
      A tuple containing roll, pitch, and yaw angles in radians.
      If input is (3,), output is (3,). If input is (N, 3), output is (N, 3).
    """
    # Normalize the direction vector(s)
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    d_x, d_y, d_z = direction[..., 0], direction[..., 1], direction[..., 2]
    # Calculate yaw (rotation around Z-axis)
    yaw = np.arctan2(d_y, d_x)
    # Calculate pitch (rotation around Y-axis)
    pitch = np.arctan2(np.sqrt(d_x**2 + d_y**2), d_z)
    # Roll is typically zero in a direction-only alignment
    roll = np.zeros_like(yaw)

    return np.stack([roll, pitch, yaw], axis=-1)


def calculate_box_inertia(mass, size_x, size_y, size_z):
    """
    Calculates the inertia tensor components for a solid box
    centered at the origin with uniform density.
    """
    ixx = (1/12.0) * mass * (size_y**2 + size_z**2)
    iyy = (1/12.0) * mass * (size_x**2 + size_z**2)
    izz = (1/12.0) * mass * (size_x**2 + size_y**2)
    # Off-diagonal terms are zero for symmetry if axes align with box edges
    ixy = 0.0
    ixz = 0.0
    iyz = 0.0
    return ixx, iyy, izz, ixy, ixz, iyz

def compute_energy(x: np.ndarray):

    n = x.shape[0]
    points = x.copy().reshape((-1, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)  # project to sphere

    # Pairwise differences
    # (n, 1, 3) - (1, n, 3) -> (n, n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Upper triangle mask to avoid double-counting and division by zero
    i, j = np.triu_indices(n, k=1)
    return np.sum(1.0 / dist[i, j])
    

def satisfy_min_distance(x: np.ndarray, min_distance: float = 0.05):
    """check if all points are at least min_distance apart"""
    n = x.shape[0]
    points = x.copy().reshape((-1, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)  # project to sphere

    # Pairwise differences
    # (n, 1, 3) - (1, n, 3) -> (n, n, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(n, k=1)
    return np.all(dist[i, j] > min_distance)

# https://math.stackexchange.com/questions/3827839/spherical-coordinates-of-a-regular-dodecahedron

center = np.array([0,0,0])

angle_top = np.arctan((3+np.sqrt(5))/4)
angle_low = np.arctan((3-np.sqrt(5))/4)

z_top = np.sin(angle_top)
r_top = np.cos(angle_top)

z_low = np.sin(angle_low)
r_low = np.cos(angle_low)

# print(z_top, z_low)
# print(r_top, r_low)

t = np.arange(0, 2*np.pi, np.pi*2/5)
t2 = t + np.pi/5
top_points = np.column_stack((r_top*np.cos(t),r_top*np.sin(t),np.ones(5)*z_top))
top_mid_points = np.column_stack((r_low*np.cos(t),r_low*np.sin(t),np.ones(5)*z_low))
low_mid_points = np.column_stack((r_low*np.cos(t2),r_low*np.sin(t2),np.ones(5)*-z_low))
low_ponts = np.column_stack((r_top*np.cos(t2),r_top*np.sin(t2),np.ones(5)*-z_top))

vertices = np.concatenate((top_points, top_mid_points, low_mid_points, low_ponts), axis=0)#*base_radius


edges = np.array([
    # top
    [0,1],
    [1,2],
    [2,3],
    [3,4],
    [4,0],

    # # top mid
    # [5,6],
    # [6,7],
    # [7,8],
    # [8,9],
    # [9,5],
    
    # # low mid
    # [10,11],
    # [11,12],
    # [12,13],
    # [13,14],
    # [14,10],

    # low
    [15,16],
    [16,17],
    [17,18],
    [18,19],
    [19,15],

    # top to top mid
    [0,5],
    [1,6],
    [2,7],
    [3,8],
    [4,9],
    
    # top mid to low mid
    [5,10],
    [6,11],
    [7,12],
    [8,13],
    [9,14],

    # low mid to top mid
    [5,14],
    [6,10],
    [7,11],
    [8,12],
    [9,13],
    
    # low mid to low
    [10,15],
    [11,16],
    [12,17],
    [13,18],
    [14,19],
])


faces = np.array([
    [0, 1, 2, 3, 4],       #  20 Top face
    [15, 16, 17, 18, 19],  #  21 Bottom face


    [0, 5, 10, 6, 1],     # 22 top mid Side face 1
    [1, 6, 11, 7, 2],     # 23 top mid Side face 2
    [2, 7, 12, 8, 3],     # 24 top mid Side face 3
    [3, 8, 13, 9, 4],     # 25 top mid Side face 4
    [4, 9, 14, 5, 0],     # 26 top mid Side face 5

    [5, 14, 19, 15, 10],   # 27 bottom mid Side face 1
    [6, 10, 15, 16, 11],   # 28 bottom mid Side face 2
    [7, 11, 16, 17, 12],   # 29 bottom mid Side face 3
    [8, 12, 17, 18, 13],   # 30 bottom mid Side face 4
    [9, 13, 18, 19, 14],   # 31 bottom mid Side face 5
    
], dtype=int)
face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])
face_centers_extended = face_centers/np.linalg.norm(face_centers,axis=1,keepdims=True)

vertices_extended = np.concatenate((vertices, face_centers_extended), axis=0)

x_axis_candidiate = np.array([
    # top
    [0,1],  
    [1,2],
    [2,3],
    [3,4],
    [4,0],
    # mid-top
    [5,0],
    [6,1],
    [7,2],
    [8,3],
    [9,4],
    # mid-low
    [10,15],
    [11,16],
    [12,17],
    [13,18],
    [14,19],
    # low
    [15,16],
    [16,17],
    [17,18],
    [18,19],
    [19,15],

    # rest are extended face centers
    [20, 0], # top
    [21,15], # bottom
    # side top face centers
    [22,23],
    [23,24],
    [24,25],
    [25,26],
    [26,22],
    # side bottom face centers
    [27,28],
    [28,29],
    [29,30],
    [30,31],
    [31,27],
])


def compute_joint_rotation(vertices):
    joint_rotation = np.zeros([len(vertices), 3, 3])

    
    for k in range(len(vertices)):
        z_axis = vertices[k] - center
        z_axis/=np.linalg.norm(z_axis)

        y_axis = np.cross(z_axis, vertices[x_axis_candidiate[k,1]] - vertices[x_axis_candidiate[k,0]])
        y_axis/=np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis/=np.linalg.norm(x_axis)

        joint_rotation[k] = np.column_stack((x_axis, y_axis, z_axis)) # x_axis, y_axis, z_axis

    rpys = R.from_matrix(joint_rotation).as_euler('xyz')

    edge_directions =vertices[edges[:,1]] - vertices[edges[:,0]]
    edge_directions/=np.linalg.norm(edge_directions,axis=1,keepdims=True)
    edge_rpy = vector_to_rpy(edge_directions)
    edge_center = (vertices[edges[:,0]] + vertices[edges[:,1]])/2
    return rpys, joint_rotation, edge_directions, edge_rpy,edge_center

rpys, joint_rotation, edge_directions, edge_rpy,edge_center = compute_joint_rotation(vertices_extended)


def sample_points_in_unit_disk_polar_batch(num_points):
    """
    Samples a batch of points uniformly within a unit disk using polar coordinates.

    Args:
        num_points (int): The number of points to sample.

    Returns:
        numpy.ndarray: A 2D NumPy array of shape (num_points, 2), where each row
                       represents an (x, y) coordinate of a sampled point.
    """
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")
    # Generate a batch of random radius squared values uniformly between 0 and 1.
    # Taking the square root ensures a uniform area distribution.
    r_squared = np.random.uniform(0, 1, num_points)
    r = np.sqrt(r_squared)
    # Generate a batch of random angles uniformly between 0 and 2*pi
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    # Convert polar coordinates to Cartesian coordinates for the entire batch
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Combine x and y coordinates into a single array of shape (num_points, 2)
    points = np.stack((x, y), axis=-1)

    return points

from dataclasses import dataclass
@dataclass
class RigidBody:
    mass: float
    com: np.ndarray
    inertia: np.ndarray # inertia that is expressed about the local COM in the local frame
    transform: np.ndarray = np.eye(4)

def compute_compound_inertia(rigid_bodies):
    total_mass = sum(rb.mass for rb in rigid_bodies)
    total_com = sum(rb.mass * rb.com for rb in rigid_bodies) / total_mass
    total_inertia = np.zeros((3, 3))
    for rb in rigid_bodies:
        r = rb.com - total_com
        inertia = rb.inertia + rb.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        total_inertia += inertia
    return RigidBody(total_mass, total_com, total_inertia)

def get_inertia(ixx, ixy, ixz, iyy, iyz, izz):
    inertia = np.zeros((3, 3))
    inertia[0, 0] = ixx
    inertia[0, 1] = ixy
    inertia[0, 2] = ixz
    inertia[1, 0] = ixy
    inertia[1, 1] = iyy
    inertia[1, 2] = iyz
    inertia[2, 0] = ixz
    inertia[2, 1] = iyz
    inertia[2, 2] = izz
    return inertia


def compute_compound_inertia_vectorized(rigid_bodies) -> RigidBody:
    """
    Computes the total mass, center of mass, and inertia tensor of a compound
    rigid body composed of multiple individual rigid bodies with arbitrary
    transforms, using vectorized operations for efficiency. The resulting
    inertia tensor is expressed about the center of mass of the compound body
    in the world frame.

    Args:
        rigid_bodies: A list of RigidBody objects, each with mass, local COM,
                      local inertia tensor, and a world transform.

    Returns:
        A new RigidBody object representing the compound body with its total
        mass, center of mass in the world frame, and inertia tensor about its
        world-frame center of mass.
    """
    num_bodies = len(rigid_bodies)
    if num_bodies == 0:
        return RigidBody(0.0, np.zeros(3), np.zeros((3, 3)), np.eye(4))

    masses = np.array([rb.mass for rb in rigid_bodies])
    local_coms = np.array([rb.com for rb in rigid_bodies])
    transforms = np.array([rb.transform for rb in rigid_bodies])
    local_inertias = np.array([rb.inertia for rb in rigid_bodies])

    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]
    # print(rotations.shape, translations.shape)

    total_mass = np.sum(masses)
    world_coms  = np.einsum('ijk,ik->ij', rotations, local_coms) + translations
    weighted_world_coms = world_coms * masses[:, None]
    total_world_com = np.sum(weighted_world_coms, axis=0) / total_mass

    total_inertia_world = np.zeros((3, 3))
    for i in range(num_bodies):
        R = transforms[i, :3, :3]
        local_inertia = local_inertias[i]
        mass = masses[i]
        world_rb_com = world_coms[i] # COM of body i in world frame

        # 1. Rotate local inertia tensor to world frame orientation
        # This inertia is about the individual body's COM (world_rb_com)
        inertia_world_frame_about_rb_com = R @ local_inertia @ R.T

        # 2. Calculate the shift vector 'd' from the total COM to the individual body's COM
        d = world_rb_com - total_world_com

        # 3. Apply Parallel Axis Theorem to shift inertia from individual COM (world_rb_com)
        #    to the total compound COM (total_world_com).
        #    PAT: I_new = I_com + m * ( (d.d)I - d outer d )
        inertia_shift_term = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        inertia_contribution_about_total_com = inertia_world_frame_about_rb_com + inertia_shift_term

        # 4. Sum the contributions
        total_inertia_world += inertia_contribution_about_total_com

    return RigidBody(total_mass, total_world_com, total_inertia_world, np.eye(4)) # Keep world transform as identity for the compound body representation


R02 = 2
R03 = 3

@dataclass
class Args():
    dof: int = 20
    add_load: bool = False
    use_miminum: bool = True
    use_motor_type: int = R02

args = Args(
    dof = 20,
    # dof = 32,
    # dof = 12,

    add_load=False,
    # add_load=True,

    # use_miminum=False,
    use_miminum=True,

    # CHOOSE MOTOR TYPE
    use_motor_type = R02
    # use_motor_type = R03
)



base_radius = 0.63/2
actuator_length=0.3
actuator_extent=[-0.105, 0.105] # actuation range
actuator_radius=0.003

end_sphere_radius=0.06
actuator_offset=0.21

anchor_point_offset = 0.01

leg_mass=0.3 # [kg]


dof = args.dof

robot_name=f"argus_dof{dof}"
if args.use_miminum:
    robot_name += "_minimum"

if args.add_load:
    robot_name += "_load"

idx_start_end = { # dof -> idx start, end
    12: (20,32),
    20: (0,20),
    32: (0,32)
}


vertices_to_use = vertices_extended[idx_start_end[dof][0]:idx_start_end[dof][1]]
rpys_to_use = rpys[idx_start_end[dof][0]:idx_start_end[dof][1]]
joint_rotations_to_use = joint_rotation[idx_start_end[dof][0]:idx_start_end[dof][1]]

base_mass_single_actuator_sturctural_part = 0.25

max_velocity=2
drum_radius=0.04

if args.use_motor_type == R03:
    base_mass_single_actuator_motor = 0.9 # robstride 03 motor
    robot_name += "_robstride03"
    max_effort = 55/drum_radius
if args.use_motor_type == R02:
    base_mass_single_actuator_motor = 0.42 # robstride 02 motor
    # robot_name += "robstride02"
    max_effort = 15/drum_radius

rb_base_link_no_actuator = RigidBody(
    mass=3.6, 
    com=np.array([0, 0, -0.065]), # TODO
    # com=np.array([0, 0, 0])
    inertia=get_inertia(ixx=0.09, ixy=0, ixz=0, iyy=0.09, iyz=0, izz=0.025)
)

rb_single_actuator_main = RigidBody(
    mass=base_mass_single_actuator_sturctural_part+ base_mass_single_actuator_motor, 
    com=np.array([-0.044, 0.0092, 0.0081]),
    # com=np.array([0, 0, 0]),
    inertia=get_inertia(ixx=0.00044, ixy=0, ixz=0, iyy=0.0008, iyz=0, izz=0.0007)
)

dof_start = idx_start_end[dof][0]
dof_end = idx_start_end[dof][1]
rb_single_actuator_main_list = []
for k in range(dof):
    rb_single_actuator_main_k = copy.deepcopy(rb_single_actuator_main)
    rb_single_actuator_main_k.transform[:3, 3] = vertices_to_use[k]
    rb_single_actuator_main_k.transform[:3, :3] = joint_rotations_to_use[k]
    rb_single_actuator_main_list.append(rb_single_actuator_main)


rb_base_link = compute_compound_inertia_vectorized(rb_single_actuator_main_list+ [rb_base_link_no_actuator])


base_mass = rb_base_link_no_actuator.mass + rb_single_actuator_main.mass * dof
print(f"base_mass: {base_mass}")
total_mass = base_mass + leg_mass * dof
print(f"total_mass: {total_mass}")
print(f"base inertia: {rb_base_link.inertia}")


#########
material_white = Material("white", Color(rgba="1 1 1 1"))
material_red = Material("red", Color(rgba="1 0 0 1"))
material_grey = Material("grey", Color(rgba="0.5 0.5 0.5 1"))

import matplotlib.pyplot as plt
cmap = plt.get_cmap("viridis")
materials = [
    Material(f"color_{i}", Color(" ".join(map(str, cmap(i/dof)[:3] + (1.0,)))))
    for i in range(20)
]


# save_path = f"{os.path.dirname(os.path.abspath(__file__))}/urdf/{robot_name}"
save_path = f"{os.getcwd()}/urdf/argus"
os.makedirs(save_path, exist_ok=True)


# base_link_visual_mesh = "simple_meshes/base_link.stl"
base_link_visual_mesh = "simple_meshes/base_link_inner.stl"

# base_link_collision_mesh = "simple_meshes/base_link.stl"
base_link_collision_mesh = "meshes/base_link.stl"


base_link_geometry = [
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Geometry(
            # Mesh(filename="base_link_static.stl", scale=[1, 1, 1])  # Reference the saved mesh
            Mesh(filename=base_link_visual_mesh, scale=[1, 1, 1])  # Reference the saved mesh
        )
]

base_link = Link(
    name="base_link"
)

if args.use_motor_type==R03:
    base_link.append(
    Inertial(
        # robstride 03
        Mass(value=24.2),
        Origin(xyz=[0, 0, -0.0038], rpy=[0, 0, 0]),
        Inertia(
            ixx=1.6, ixy=0, ixz=0,
            iyy=1.6, iyz=0, izz=1.74
        ),
        ),
    )
elif args.use_motor_type==R02: # robstride 02
    base_link.append(
        Inertial(
            # robstride 02
            Mass(value=base_mass),
            # # for real robot
            # Origin(xyz=[0, 0, -0.01045], rpy=[0, 0, 0]),
            # Inertia(
            #     ixx=0.9, ixy=0, ixz=0,
            #     iyy=0.9, iyz=0, izz=0.95
            # ),

            # for simulation only
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Inertia(
                ixx=0.95, ixy=0, ixz=0,
                iyy=0.95, iyz=0, izz=0.95
            ),
        ),
    )

edge_radius = 0.003
edge_length = 0.15


approximate_base_as_ball = False

if approximate_base_as_ball:
    base_sphere = [
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Geometry(Sphere(radius=0.1))]
    base_link.append(Visual(*base_sphere, material_grey))
else:
    base_link.append(Visual(*base_link_geometry, material_grey))
    # # Collision(*base_link_geometry),
    # for i in range(len(edges)):
    #     edge = edges[i]
    #     cylinder_1 = [
    #         Origin(xyz=(edge_center[i]*base_radius).tolist(), rpy=edge_rpy[i].tolist()),
    #         Geometry(Cylinder(radius=edge_radius,length=edge_length))]
    #     base_link.append(Visual(*cylinder_1))
    #     if not args.use_miminum:
    #         base_link.append(Collision(*cylinder_1))


if (not args.use_miminum):
    import ipdb; ipdb.set_trace()
    for i, (vertex, rpy) in enumerate(zip(vertices_to_use, rpys_to_use)):
        actuator_main = [ 
            Origin(xyz=(vertex * (base_radius+anchor_point_offset)).tolist(), rpy=rpy.tolist()),
            Geometry(Mesh(filename="simple_meshes/main_simple.stl", scale=[1, 1, 1]))]
        base_link.append(Visual(*actuator_main,material_grey))
        base_link.append(Collision(*actuator_main))
else:
    # pass
    for i, (vertex, rpy) in enumerate(zip(vertices_to_use, rpys_to_use)):
        actuator_main = [ 
            Origin(xyz=(vertex * (base_radius+anchor_point_offset)).tolist(), rpy=rpy.tolist()),
            Geometry(Mesh(filename="simple_meshes/main_simple.stl", scale=[1, 1, 1]))]
        base_link.append(Visual(*actuator_main, material_white))


cylinder_radius = 0.003
cylinder_separation_radius = 0.021

np.random.seed(42) # for reproducibility

# rpys, joint_rotation, edge_directions, edge_rpy,edge_center = compute_joint_rotation(vertices_extended)






vertices_list = []

# num_variants = 32
num_variants = 512
half_num_variants = num_variants//2





from symmetry_generation import compute_mimum_energy_vertices
vertices_minimum_energy = compute_mimum_energy_vertices(n=20)
min_energy = compute_energy(vertices_minimum_energy[:dof])

def generate_varients(vertices,num_pre=half_num_variants*100,num_select=half_num_variants, energy_range = [min_energy, min_energy*1.5]):
    pre_vertices_list = []
    pre_energy_list = []
    for k in range(num_pre):
        perturbation = np.zeros((len(vertices),3))
        perturbation[:,:2] = sample_points_in_unit_disk_polar_batch(len(vertices))
        perturbation_scale = ((k%num_pre)/num_pre)**0.5
        vertices_to_use_new = (vertices + perturbation*perturbation_scale)
        vertices_to_use_new/= np.linalg.norm(vertices_to_use_new, axis=1, keepdims=True) # normalize to unit spheres
        pre_vertices_list.append(vertices_to_use_new)
        pre_energy_list.append(compute_energy(vertices_to_use_new[:dof]))
    pre_energy_list = np.array(pre_energy_list)
    pre_vertices_list = np.stack(pre_vertices_list)

    index_in_liers = pre_energy_list < energy_range[1]

    index_in_liers = index_in_liers * np.array([satisfy_min_distance(v) for v in pre_vertices_list], dtype=bool)

    pre_energy_list = pre_energy_list[index_in_liers]
    pre_vertices_list = pre_vertices_list[index_in_liers]
    
    num_bins = 20
    hist, bin_edges = np.histogram(pre_energy_list, bins=num_bins)
    bin_indices = np.digitize(pre_energy_list, bin_edges[:-1], right=False)

    # Compute inverse frequency weights
    bin_counts = hist[bin_indices - 1]
    weights = 1.0 / bin_counts

    # Normalize weights
    weights /= weights.sum()

    # Sample without replacement
    indices = np.random.choice(len(pre_energy_list), size=num_select, replace=False, p=weights)

    # # plt.hist(pre_energy_list, bins=10)
    # plt.hist(pre_energy_list[indices], bins=num_bins)
    # plt.xlabel('Energy')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Energy')
    # plt.show()

    return pre_vertices_list[indices], pre_energy_list[indices]

pre_vertices_list_varient_1, pre_energy_list_varient_1 = generate_varients(vertices_to_use)
pre_vertices_list_varient_2, pre_energy_list_varient_2 = generate_varients(vertices_minimum_energy)

# data = pre_energy_list_varient_1
# vertices_list = pre_vertices_list_varient_1

# max_energy_threshold = min_energy*2
# index_in_liers = data < max_energy_threshold

# vertices_list_in_liers = vertices_list[index_in_liers]

# hist, bin_edges = np.histogram(data, bins=10, range = (min_energy, min_energy*2))
# bin_indices = np.digitize(data, bin_edges[:-1], right=False)


for k in range(num_variants):

    group = Group()
    # group(material_red)
    # group(material_white)
    group(*materials)
    group(base_link)

    # perturbation = np.zeros((len(vertices_to_use),3))
    # perturbation[:,:2] = sample_points_in_unit_disk_polar_batch(len(vertices_to_use))

    # perturbation_scale = ((k%half_num_variants)/half_num_variants)**0.5
    # print(perturbation_scale)
    
    if k == 0:
        vertices_to_use_new = vertices_to_use.copy()
    elif k < half_num_variants:
        # vertices_to_use_new = (vertices_to_use + perturbation*perturbation_scale)
        # vertices_to_use_new/= np.linalg.norm(vertices_to_use_new, axis=1, keepdims=True) # normalize to unit spheres
        vertices_to_use_new = pre_vertices_list_varient_1[k%half_num_variants]
    elif k < num_variants-1:
        # vertices_to_use_new = vertices_minimum_energy + perturbation*perturbation_scale
        # vertices_to_use_new/= np.linalg.norm(vertices_minimum_energy, axis=1, keepdims=True) # normalize to unit spheres
        vertices_to_use_new = pre_vertices_list_varient_2[k%half_num_variants]
    else:
        vertices_to_use_new = vertices_minimum_energy

    rpys_new, joint_rotation_new, edge_directions_new, edge_rpy_new,edge_center_new = compute_joint_rotation(vertices_to_use_new)


    # for i, vertex in enumerate(vertices):
    # for i, (vertex, rpy) in enumerate(zip(vertices_to_use, rpys_to_use)):
    for i, (vertex, rpy) in enumerate(zip(vertices_to_use_new, rpys_new)):


        joint_origin = vertex * (base_radius+actuator_offset+end_sphere_radius)
        

        end_sphere = [
            Origin(xyz=[0, 0, -end_sphere_radius], rpy=[0, 0, 0]),
            Geometry(Sphere(radius=end_sphere_radius)) ]


        link = Link(
            # Visual(
            #     Origin(xyz=[0, 0, -0.5*actuator_length+actuator_offset], rpy=[0, 0, 0]),
            #     Geometry(Cylinder(radius=actuator_radius,length=actuator_length))),
            # Collision(
            #     Origin(xyz=[0, 0, -0.5*actuator_length+actuator_offset], rpy=[0, 0, 0]),
            #     Geometry(Cylinder(radius=actuator_radius,length=actuator_length))),

            # Visual(*end_sphere, material_red) if i==0 else Visual(*end_sphere, material_white),
            Visual(*end_sphere, materials[i%dof]),

            
            Collision(*end_sphere),

            Inertial(
                Origin(xyz=[-0.00073, 0, 0.15817], rpy=[0, 0, 0]),
                Mass(value=leg_mass),
                # # TODO: calculate inertia for real
                # Inertia(ixx=leg_mass*(3*actuator_radius**2+actuator_length**2)/12, ixy=0, ixz=0,
                #         iyy=leg_mass*(3*actuator_radius**2+actuator_length**2)/12, iyz=0, izz=leg_mass*actuator_radius**2/2),
                Inertia(ixx=0.0035, ixy=0, ixz=0,
                        iyy=0.0035, iyz=0, izz=0.00043),
            ),
            name=f"link_{i}",
        )

        support_cylinders = [
            [Origin(xyz=[cylinder_separation_radius*np.cos(i*2*np.pi/3), cylinder_separation_radius*np.sin(i*2*np.pi/3), -actuator_offset-end_sphere_radius+anchor_point_offset], rpy=[0, 0, 0]),
            Geometry(Cylinder(radius=cylinder_radius,length=actuator_length))]
            for i in range(3)
        ]

        for cylinder in support_cylinders:
            link.append(Visual(*cylinder))
            if not args.use_miminum:
                link.append(Collision(*cylinder))



        joint = Joint(
            Parent(base_link),
            Child(link),
            Origin(xyz=joint_origin.tolist(), rpy=rpy.tolist()),
            Axis(xyz=[0, 0, 1]),
            Limit(effort=max_effort, velocity=max_velocity, lower=actuator_extent[0], upper=actuator_extent[1]),
            type="prismatic",
            name=f"actuator_{i:03d}_joint",
        )
        group(joint)

        group(link)

    if args.add_load:
        # add cube to the base link with fixed mass
        cube_size=[0.1, 0.1, 0.1] # meters (x, y, z)
        cube_mass=1.0             # kilograms

        cube = [
                Origin(xyz=[0,0,0], rpy=[0, 0, 0]),
                Geometry(Box(size=cube_size))
                ]
        cube_link  = Link(
            Visual(*cube),
            Collision(*cube),
            Inertial(
                Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
                Mass(value=cube_mass),
                Inertia(*calculate_box_inertia(cube_mass, cube_size[0], cube_size[1], cube_size[2])),
            ),
            name="cube",
        )

        cube_joint = Joint(
            Parent(base_link),
            Child(cube_link),
            Origin(xyz=(vertices_extended[20] * (base_radius+anchor_point_offset)).tolist(), rpy=rpys[20].tolist()),
            type="fixed",
            name="cube_joint",
        )
        group(cube_link)
        group(cube_joint)

        group(Material(Color(rgba=[1, 0, 0, 1]),name="red"))

    robot = Robot(
        group,
        name='argus'
    )

    # urdf_path = os.path.abspath(f"{save_path}/{robot_name}.urdf")

    vertices_list.append(vertices_to_use_new)
   

    urdf_path = os.path.abspath(f"{save_path}/sim_rand_joint_{k:04d}_{robot_name}.urdf")

    # print(robot.urdf())
    with open(urdf_path, "w") as f:
        f.write(robot.urdf())
        print(f"written to {urdf_path}")
    
    # break

all_energy = []
for k in range(num_variants):
    energy = compute_energy(vertices_list[k][:dof])
    all_energy.append(energy)
    # print(f"Energy of the configuration: {energy}")
    
np.save(f"{save_path}/energy.npy", all_energy)

# plot the distribution of energy
plt.hist(all_energy, bins=10)
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.title('Distribution of Energy')
plt.show()


