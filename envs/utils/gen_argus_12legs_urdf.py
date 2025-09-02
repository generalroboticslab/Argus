import numpy as np
from odio_urdf import *
import os


def icosahedron_vertices():
    """
    Calculates the vertices of an icosahedron with edge length 2.

    Returns:
      A NumPy array of shape (12, 3) representing the vertices.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a unit icosahedron centered at the origin
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])

    vertices = vertices/np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices

def icosahedron_triangle_centers(vertices):
    """
    Calculates the centers of the triangles of an icosahedron.

    Args:
        vertices: A NumPy array of shape (12, 3) representing the vertices.

    Returns:
        A NumPy array of shape (20, 3) representing the triangle centers.
    """

    # Triangular faces of the icosahedron (each row represents a triangle by vertex indices)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    # Calculate the centers of each face
    centers = np.array([
        vertices[face].mean(axis=0) for face in faces
    ])
    # Normalize to keep the centers on the icosahedron surface if needed
    centers /= np.linalg.norm(centers, axis=1)[:, np.newaxis]
    return centers


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
        sphere_radius=0.25,
        actuator_length=0.65, # length of cylinder bar
        actuator_extent=0.4, # actuation range
        actuator_radius=0.04,
        actuator_offset=-0.15,
        end_sphere_radius=0.06,
        max_effort=300,
        max_velocity=2.1,
        leg_mass=0.3,
        base_mass=10,
        robot_name="12legs_argus_length_40"
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

    joint_origin_extent = actuator_length+actuator_offset
    vertices = icosahedron_vertices() * joint_origin_extent

    camera_rpy = vector_to_rpy(vertices)
    camera_quat = get_quaternion_from_euler(camera_rpy)
    camera_quat = quaternion_multiply_batch(camera_quat, np.array([0, -0.7071068, 0, 0.7071068]*len(camera_quat)))
    np.save( f"{save_path}/camera_p.npy",np.array(vertices))
    np.save( f"{save_path}/camera_r.npy",np.array(camera_quat))

    # Calculate the triangle centers
    # triangle_centers = icosahedron_triangle_centers(vertices)*joint_origin_extent
    # vertices = triangle_centers

    rpys = vector_to_rpy(vertices)

    base_link = Link(
        Visual(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Geometry(
                Sphere(radius=sphere_radius)),
        ),
        Collision(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Geometry(
                Sphere(radius=sphere_radius)),
        ),
        Inertial(
            Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
            Mass(value=base_mass),
            Inertia(ixx=(2*base_mass*sphere_radius**2)/5, ixy=0, ixz=0, iyy=(2*base_mass*sphere_radius**2)/5, iyz=0, izz=(2*base_mass*sphere_radius**2)/5),
        ),
        name="base_link",
    )

    group = Group(base_link)

    for i, vertex in enumerate(vertices):
        rpy = rpys[i]
        link = Link(
            Visual(
                Origin(xyz=[0, 0, -0.5*actuator_length], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius,
                                  length=actuator_length)),
            ),
            Visual(
                Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
                Geometry(Sphere(radius=end_sphere_radius)),
            ),
            Collision(
                Origin(xyz=[0, 0, -0.5*actuator_length], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius,
                                  length=actuator_length)),
            ),
            Collision(
                Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
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
        f"{save_path}/argus_12legs_urdf.urdf")

    # Save the URDF to a file (optional)
    with open(save_path, "w") as f:
        f.write(robot.urdf())

generate_urdf()
