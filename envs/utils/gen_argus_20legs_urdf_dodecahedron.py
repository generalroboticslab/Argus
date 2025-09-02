import trimesh
import numpy as np
import numpy as np
from odio_urdf import *
import os


def scale_to_diameter(vertices, radius):
    # Calculate the current radius (distance from origin to a vertex)
    current_radius = np.linalg.norm(vertices[0])  # Assuming all vertices are at the same distance
    scale_factor = radius / current_radius  # Diameter is twice the radius

    # Scale all vertices
    return vertices * scale_factor

def generate_dodecahedron(radius,save_path):

    # Define the 20 vertices of a regular dodecahedron (from the VTK example)
    vertices = np.array([
        [1.21412, 0, 1.58931],
        [0.375185, 1.1547, 1.58931],
        [-0.982247, 0.713644, 1.58931],
        [-0.982247, -0.713644, 1.58931],
        [0.375185, -1.1547, 1.58931],
        [1.96449, 0, 0.375185],
        [0.607062, 1.86835, 0.375185],
        [-1.58931, 1.1547, 0.375185],
        [-1.58931, -1.1547, 0.375185],
        [0.607062, -1.86835, 0.375185],
        [1.58931, 1.1547, -0.375185],
        [-0.607062, 1.86835, -0.375185],
        [-1.96449, 0, -0.375185],
        [-0.607062, -1.86835, -0.375185],
        [1.58931, -1.1547, -0.375185],
        [0.982247, 0.713644, -1.58931],
        [-0.375185, 1.1547, -1.58931],
        [-1.21412, 0, -1.58931],
        [-0.375185, -1.1547, -1.58931],
        [0.982247, -0.713644, -1.58931]
    ])

    # Define the faces of the dodecahedron (from the VTK example)
    faces = [
        [0, 1, 2, 3, 4],    # Face 1
        [0, 5, 10, 6, 1],    # Face 2
        [1, 6, 11, 7, 2],    # Face 3
        [2, 7, 12, 8, 3],    # Face 4
        [3, 8, 13, 9, 4],    # Face 5
        [4, 9, 14, 5, 0],    # Face 6
        [15, 10, 5, 14, 19], # Face 7
        [16, 11, 6, 10, 15], # Face 8
        [17, 12, 7, 11, 16], # Face 9
        [18, 13, 8, 12, 17], # Face 10
        [19, 14, 9, 13, 18], # Face 11
        [19, 18, 17, 16, 15] # Face 12
    ]

    # Scale vertices to the desired diameter
    scaled_vertices = scale_to_diameter(vertices, radius)
    
    # Create the dodecahedron mesh with the scaled vertices
    dodecahedron = trimesh.Trimesh(vertices=scaled_vertices, faces=faces, process=True)
    # Repair face normals to ensure they are consistent
    trimesh.repair.fix_normals(dodecahedron)
    # Save the mesh to a file (OBJ format)
    file_path = f"{save_path}/mesh/dodecahedron_mesh.obj"
    os.makedirs(f"{save_path}/mesh", exist_ok=True)

    dodecahedron.export(file_path)
    print(f"Mesh saved to {file_path}")
    # Show the mesh with face normals for debugging
    # dodecahedron.show(face_normals=True)

    return scaled_vertices

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

def get_quaternion_from_euler(rpy):
    """
    Convert Euler angles to quaternions.

    Input:
        :param rpy: A single set of roll, pitch, yaw angles as a 1D array [roll, pitch, yaw]
                    or a 2D array of shape (N, 3) where N is the number of sets of angles.

    Output:
        :return: A quaternion [x, y, z, w] for each input set of angles.
                 If the input is (N, 3), the output is an array of shape (N, 4).
    """
    rpy = np.atleast_2d(rpy)  # Ensure the input is 2D (N, 3)

    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    quaternions = np.column_stack((qx, qy, qz, qw))  # Stack the results along axis 1
    return quaternions if quaternions.shape[0] > 1 else quaternions[0]

def quaternion_multiply_batch(q1, q2):
    """
    Multiply two batches of quaternions, or a single quaternion with a batch.

    :param q1: A single quaternion [x1, y1, z1, w1] or a batch of quaternions of shape (N, 4).
    :param q2: A batch of quaternions of shape (N, 4) to which q1 is applied.
    :return: Resultant batch of quaternions of shape (N, 4).
    """
    q1 = np.atleast_2d(q1)  # Ensure q1 is at least 2D
    q2 = np.atleast_2d(q2)  # Ensure q2 is at least 2D

    if q1.shape[0] == 1:
        q1 = np.tile(q1, (q2.shape[0], 1))  # Broadcast q1 to match q2's batch size

    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    xr = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    yr = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    zr = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    wr = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.column_stack((xr, yr, zr, wr))


def generate_urdf(
        base_radius=0.30,
        actuator_length=0.34, # length of cylinder bar
        actuator_extent=0.2, # actuation range
        actuator_radius=0.03,
        actuator_offset=0.03+0.08, # offset from the center of the base_link 0.6 + foot node 0.04
        end_sphere_radius=0.05,
        max_effort=300,
        max_velocity=2,
        leg_mass=0.3,
        base_mass=17.5,
        robot_name="20legs_dodecahedron_small"
):
    """
    Generates a URDF string with a sphere base_link and 12 linear actuators.

    Args:
      sphere_radius: Radius of the base_link sphere.
      actuator_length: Length of each linear actuator.

    Returns:
      A string containing the URDF XML.
    """
    save_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/urdf/{robot_name}"
    os.makedirs(save_path, exist_ok=True)

    vertices = generate_dodecahedron(base_radius,save_path) 

    camera_rpy = vector_to_rpy(vertices)
    camera_quat = get_quaternion_from_euler(camera_rpy)
    camera_quat = quaternion_multiply_batch(camera_quat, np.array([0, -0.7071068, 0, 0.7071068]*len(camera_quat)))
    np.save( f"{save_path}/camera_p.npy",np.array(vertices))
    np.save( f"{save_path}/camera_r.npy",np.array(camera_quat))

    rpys = vector_to_rpy(vertices)
    base_link = Link(
        Visual(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Geometry(
                Mesh(filename="mesh/dodecahedron_mesh.obj", scale=[1, 1, 1])  # Reference the saved mesh
            ),
        ),
        Collision(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Geometry(
                Mesh(filename="mesh/dodecahedron_mesh.obj", scale=[1, 1, 1])  # Reference the saved mesh
            ),
        ),
        Inertial(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Mass(value=base_mass),
            Inertia(
                ixx=(2 * base_mass * base_radius**2) / 5, ixy=0, ixz=0,
                iyy=(2 * base_mass * base_radius**2) / 5, iyz=0, izz=(2 * base_mass * base_radius**2) / 5
            ),
        ),
        name="base_link",
    )
    group = Group(base_link)

    for i, vertex in enumerate(vertices):
        rpy = rpys[i]
        link = Link(
            Visual(
                Origin(xyz=[0, 0, -0.5*actuator_length+actuator_offset], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius,
                                length=actuator_length)),
            ),
            Visual(
                Origin(xyz=[0, 0, actuator_offset], rpy=[0, 0, 0]),
                Geometry(Sphere(radius=end_sphere_radius)),
            ),
            Collision(
                Origin(xyz=[0, 0, -0.5*actuator_length+actuator_offset], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius,
                                length=actuator_length)),
            ),
            Collision(
                Origin(xyz=[0, 0, actuator_offset], rpy=[0, 0, 0]),
                Geometry(Sphere(radius=end_sphere_radius)),
            ),
            Inertial(
                Origin(xyz=[0, 0, -0.5*actuator_length], rpy=[0, 0, 0]),
                Mass(value=leg_mass),
                # TODO: calculate inertia for real
                Inertia(ixx=leg_mass*(3*actuator_radius**2+actuator_length**2)/12, ixy=0, ixz=0,
                        iyy=leg_mass*(3*actuator_radius**2+actuator_length**2)/12, iyz=0, izz=leg_mass*actuator_radius**2/2),
            ),
            name=f"link_{i}",
        )
        joint = Joint(
            Parent(base_link),
            Child(link),
            Origin(xyz=[vertex[0], vertex[1], vertex[2]],
                rpy=[rpy[0], rpy[1], rpy[2]]),
            Axis(xyz=[0, 0, 1]),
            Limit(effort=max_effort, velocity=max_velocity, lower=0, upper=actuator_extent),
            type="prismatic",
            name=f"actuator_{i}_joint",
        )
        group(joint)

        group(link)

    robot = Robot(
        group,
        name='argus'
    )
    print(robot.urdf())

    save_path = os.path.abspath(
        f"{save_path}/argus_dodecahedron.urdf")

    # Save the URDF to a file (optional)
    with open(save_path, "w") as f:
        f.write(robot.urdf())

generate_urdf()
