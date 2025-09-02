import numpy as np
from odio_urdf import *
import os

def rhombic_triacontahedron_vertices():
    """
    Calculates the 32 vertices of a regular rhombic triacontahedron.

    Returns:
        A NumPy array of shape (32, 3) representing the vertices.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of a dodecahedron
    dodecahedron_vertices = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        [ 0,  phi,  1/phi], [ 0, phi, -1/phi], [ 0, -phi, 1/phi], [ 0, -phi, -1/phi]
    ])

    # 20 vertices of an icosahedron face centers
    icosahedron_face_centers = np.array([
        [ phi,  0,  1], [-phi,  0,  1], [ phi,  0, -1], [-phi,  0, -1],
        [ 1,  phi,  0], [-1,  phi,  0], [ 1, -phi,  0], [-1, -phi,  0],
        [ 0,  1,  phi], [ 0, -1,  phi], [ 0,  1, -phi], [ 0, -1, -phi],
        [ phi,  1,  0], [-phi,  1,  0], [ phi, -1,  0], [-phi, -1,  0],
        [ 1/phi,  0,  phi], [-1/phi,  0,  phi], [ 1/phi,  0, -phi], [-1/phi,  0, -phi]
    ])

    # Combine and normalize vertices
    vertices = np.vstack((dodecahedron_vertices, icosahedron_face_centers))
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices

def rhombic_triacontahedron_face_centers(vertices):
    """
    Calculates the centers of the rhombic triacontahedron faces.

    Args:
        vertices: A NumPy array of shape (32, 3) representing the vertices.

    Returns:
        A NumPy array of shape (30, 3) representing the face centers.
    """
    # Define the rhombic faces by indices of vertices
    faces = [
        [0, 8, 16, 4], [0, 4, 20, 12], [0, 12, 24, 8], [1, 9, 17, 5],
        [1, 5, 21, 13], [1, 13, 25, 9], [2, 10, 18, 6], [2, 6, 22, 14],
        [2, 14, 26, 10], [3, 11, 19, 7], [3, 7, 23, 15], [3, 15, 27, 11],
        [4, 16, 28, 20], [5, 17, 29, 21], [6, 18, 30, 22], [7, 19, 31, 23],
        [8, 24, 30, 16], [9, 25, 31, 17], [10, 26, 28, 18], [11, 27, 29, 19],
        [12, 20, 28, 24], [13, 21, 29, 25], [14, 22, 30, 26], [15, 23, 31, 27]
    ]
    
    # Calculate centers by averaging the vertices for each face
    centers = np.array([vertices[face].mean(axis=0) for face in faces])

    # Normalize to keep centers on the surface
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
        actuator_length=0.65,
        actuator_extent=0.40,
        actuator_radius=0.04,
        actuator_offset=-0.15,
        end_sphere_radius=0.06,
        max_effort=500,
        max_velocity=0.75,
        leg_mass=0.3,
        base_mass=22.4,
        robot_name="rhombic_triacontahedron_40cm"
):
    """
    Generates a URDF string with a sphere base_link and 30 actuators.

    Args:
      sphere_radius: Radius of the base_link sphere.
      actuator_length: Length of each linear actuator.
        current_body_length: 0.56 m
    Returns:
      A string containing the URDF XML.
    """
    save_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/urdf/{robot_name}"
    os.makedirs(save_path, exist_ok=True)

    joint_origin_extent = actuator_length+actuator_offset
    vertices = rhombic_triacontahedron_vertices() * joint_origin_extent

    camera_rpy = vector_to_rpy(vertices)
    camera_quat = get_quaternion_from_euler(camera_rpy)
    camera_quat = quaternion_multiply_batch(camera_quat, np.array([0, -0.7071068, 0, 0.7071068]*len(camera_quat)))
    np.save( f"{save_path}/camera_p.npy",np.array(vertices))
    np.save( f"{save_path}/camera_r.npy",np.array(camera_quat))

    # Calculate the triangle centers
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
                Origin(xyz=[0, 0, -0.5 * actuator_length], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius, length=actuator_length)),
            ),
            Visual(
                Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
                Geometry(Sphere(radius=end_sphere_radius)),
            ),
            Collision(
                Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
                Geometry(Sphere(radius=end_sphere_radius)),
            ),
            Collision(
                Origin(xyz=[0, 0, -0.5 * actuator_length], rpy=[0, 0, 0]),
                Geometry(Cylinder(radius=actuator_radius, length=actuator_length)),
            ),
            Inertial(
                Origin(xyz=[0, 0, -0.5 * actuator_length], rpy=[0, 0, 0]),
                Mass(value=leg_mass),
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
        name=robot_name
    )

    print(robot.urdf())

    save_path = os.path.abspath(
        f"{save_path}/rhombic_triacontahedron_urdf.urdf")

    with open(save_path, "w") as f:
        f.write(robot.urdf())

    print(f"URDF saved to {save_path}")
generate_urdf()
