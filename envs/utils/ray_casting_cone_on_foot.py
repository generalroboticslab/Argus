import numpy as np
import trimesh

from scipy.spatial.transform import Rotation as R
rot = np.array([[-0.2746, -0.8966, -0.3247,  0.1233],
        [-0.0367, -0.6609, -0.4236,  0.6184],
        [ 0.2431, -0.9336,  0.2377,  0.1131],
        [ 0.0612, -0.6969,  0.3345,  0.6314],
        [-0.6314, -0.3345, -0.6969,  0.0612],
        [-0.1131, -0.2377, -0.9336,  0.2431],
        [ 0.6184, -0.4236,  0.6609,  0.0367],
        [ 0.1233, -0.3247,  0.8966,  0.2746],
        [-0.4173, -0.6486, -0.6099,  0.1824],
        [-0.1723, -0.5408, -0.7186,  0.4018],
        [ 0.4018, -0.7186,  0.5408,  0.1723],
        [ 0.1824, -0.6099,  0.6486,  0.4173],
        [-0.0156, -0.9861, -0.0476,  0.1586],
        [-0.6533,  0.0487, -0.7554,  0.0118],
        [ 0.0118, -0.7554, -0.0487,  0.6533],
        [-0.1586,  0.0476, -0.9861, -0.0156],
        [-0.2072, -0.7663, -0.4845,  0.3676],
        [-0.3695, -0.3971, -0.8150,  0.2038],
        [ 0.2038, -0.8150,  0.3971,  0.3695],
        [ 0.3676, -0.4845,  0.7663,  0.2072],
        [-0.5908, -0.6728, -0.4440, -0.0351],
        [ 0.5539, -0.7359,  0.3835, -0.0671],
        [ 0.0671, -0.3835, -0.7359,  0.5539],
        [-0.0351, -0.4440,  0.6728,  0.5908],
        [-0.1172, -0.8571, -0.2955,  0.4054],
        [-0.4065, -0.1992, -0.8844,  0.1134],
        [ 0.1134, -0.8844,  0.1992,  0.4065],
        [ 0.4054, -0.2955,  0.8571,  0.1172],
        [-0.0300, -0.9849, -0.0401, -0.1658],
        [-0.8609,  0.0423, -0.5063,  0.0267],
        [ 0.0267, -0.5063, -0.0423,  0.8609],
        [ 0.1658,  0.0401, -0.9849, -0.0300]])

direction = np.array([[-0.0323,  0.4905, -0.5725],
        [-0.5249,  0.4042,  0.0826],
        [-0.0770, -0.4016, -0.6933],
        [-0.4944, -0.3202,  0.0124],
        [ 0.6136,  0.3975, -0.0154],
        [ 0.0678,  0.3538,  0.6108],
        [ 0.5379, -0.4142, -0.0846],
        [ 0.0290, -0.4407,  0.5144],
        [ 0.1897,  0.6566, -0.1320],
        [-0.1158,  0.5673,  0.2204],
        [ 0.1190, -0.5830, -0.2265],
        [-0.1968, -0.6812,  0.1369],
        [-0.2221,  0.0705, -0.6746],
        [ 0.6456, -0.0380,  0.0926],
        [-0.6797,  0.0400, -0.0975],
        [ 0.2303, -0.0731,  0.6995],
        [-0.2446,  0.6037, -0.1756],
        [ 0.3321,  0.6016,  0.3102],
        [-0.3078, -0.5575, -0.2875],
        [ 0.2220, -0.5477,  0.1593],
        [ 0.4244,  0.4127, -0.4478],
        [ 0.3696, -0.3460, -0.4918],
        [-0.3805,  0.3562,  0.5063],
        [-0.3900, -0.3792,  0.4115],
        [-0.4405,  0.4235, -0.3497],
        [ 0.5033,  0.3322,  0.4409],
        [-0.4115, -0.2716, -0.3605],
        [ 0.4736, -0.4554,  0.3760],
        [ 0.2174,  0.0456, -0.6225],
        [ 0.5707,  0.0021, -0.3173],
        [-0.5340, -0.0020,  0.2969],
        [-0.2320, -0.0487,  0.6643]])
origin = np.array([[-3.3387e-02,  4.9150e-01,  1.7877e-01],
        [-5.2599e-01,  4.0519e-01,  8.3382e-01],
        [-7.8083e-02, -4.0052e-01,  5.7943e-02],
        [-4.9543e-01, -3.1918e-01,  7.6363e-01],
        [ 6.1256e-01,  3.9852e-01,  7.3586e-01],
        [ 6.6770e-02,  3.5480e-01,  1.3620e+00],
        [ 5.3686e-01, -4.1314e-01,  6.6662e-01],
        [ 2.7962e-02, -4.3970e-01,  1.2657e+00],
        [ 1.8859e-01,  6.5760e-01,  6.1925e-01],
        [-1.1690e-01,  5.6829e-01,  9.7160e-01],
        [ 1.1797e-01, -5.8195e-01,  5.2477e-01],
        [-1.9786e-01, -6.8016e-01,  8.8818e-01],
        [-2.2321e-01,  7.1531e-02,  7.6623e-02],
        [ 6.4451e-01, -3.6934e-02,  8.4380e-01],
        [-6.8081e-01,  4.1012e-02,  6.5379e-01],
        [ 2.2925e-01, -7.2060e-02,  1.4507e+00],
        [-2.4572e-01,  6.0478e-01,  5.7566e-01],
        [ 3.3103e-01,  6.0261e-01,  1.0615e+00],
        [-3.0886e-01, -5.5649e-01,  4.6372e-01],
        [ 2.2088e-01, -5.4671e-01,  9.1053e-01],
        [ 4.2333e-01,  4.1374e-01,  3.0341e-01],
        [ 3.6848e-01, -3.4494e-01,  2.5947e-01],
        [-3.8158e-01,  3.5726e-01,  1.2576e+00],
        [-3.9105e-01, -3.7819e-01,  1.1627e+00],
        [-4.4155e-01,  4.2453e-01,  4.0156e-01],
        [ 5.0224e-01,  3.3319e-01,  1.1921e+00],
        [-4.1262e-01, -2.7056e-01,  3.9076e-01],
        [ 4.7255e-01, -4.5434e-01,  1.1272e+00],
        [ 2.1633e-01,  4.6656e-02,  1.2872e-01],
        [ 5.6959e-01,  3.1313e-03,  4.3396e-01],
        [-5.3509e-01, -9.2728e-04,  1.0481e+00],
        [-2.3309e-01, -4.7652e-02,  1.4156e+00]])



def generate_rays_for_3d_conical_scan(origins=(0, 0, 0), directions=(0, 0, 1), half_angle=np.pi/4, num_rays=50,quaternions=rot):
    """
    Generate ray origins and directions for a 3D conical scan with a custom origin, direction, and half-angle.
    Rays will be uniformly distributed within the cone.

    Parameters:
        origin (tuple): The origin (x, y, z) from which rays originate.
        direction (tuple): The direction vector of the cone (x, y, z).
        half_angle (float): The half-angle of the cone in radians.
        num_rays (int): Number of rays to generate.

    Returns:
        tuple: Arrays of ray origins and ray directions.
    """
    all_ray_origins = []
    all_ray_directions = []

    for origin, direction, quaternion in zip(origins, directions, quaternions):
        # Normalize the direction vector to ensure proper cone alignment
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # Generate uniform points on the cone
        phi = np.linspace(0, np.pi/6, 5)  # Azimuthal angle
        theta = np.linspace(0, 2 * np.pi, 10)  # Polar angle
        print('phi', phi)
        print('theta', theta)
        
        # Using scipy's Rotation to rotate by x and z axis
        vector_phi_list = []
        for value in phi:
            # Rotation around the x-axis (polar angle)

            # Convert the quaternion to a Rotation object
            rotation = R.from_quat(quaternion)
            inverse_rotation = rotation.inv()
            init_vector = inverse_rotation.apply(direction)


            rotation_x = R.from_euler('x', value)  # 'x' denotes rotation around the x-axis
            rotated_vector = rotation_x.apply(init_vector)

            vector_phi_list.append(rotation.apply(rotated_vector))
        
        vector_theta_list = []
        for value in theta:
            for vector in vector_phi_list:
                rotation = R.from_quat(quaternion)
                inverse_rotation = rotation.inv()
                init_vector = inverse_rotation.apply(vector)

                rotation_z = R.from_euler('z', value)  # 'x' denotes rotation around the x-axis
                rotated_vector = rotation_z.apply(init_vector)

                vector_theta_list.append(rotation.apply(rotated_vector))

        vector_theta_list = np.array(vector_theta_list)
        ray_origins = np.tile(origin, (len(vector_theta_list), 1))  # Repeat origin for each ray
        all_ray_directions.append(vector_theta_list)
        all_ray_origins.append(ray_origins)
        
    # Concatenate all the origins and directions from different origins
    all_ray_origins = np.vstack(all_ray_origins)
    all_ray_directions = np.vstack(all_ray_directions)
    
    return all_ray_origins, all_ray_directions

# Main function to perform the 3D conical scan using ray intersections
def main():
    # Create a sample mesh (e.g., a box or any 3D object)
    plane = trimesh.primitives.Box(extents=[20, 20, 0.1], transform=trimesh.transformations.translation_matrix([0, 0, -0.1/2]))
    box = trimesh.primitives.Box(extents=[3, 1, 0.3], transform=trimesh.transformations.translation_matrix([0, 2, 1.3]))
    combined_mesh = trimesh.util.concatenate([box, plane])

    # Custom origin for ray generation (example: (1, 1, 1))
    custom_origin = (0, 0, 1.5)
    # Direction vector of the cone (e.g., pointing upwards)
    cone_direction = (0, 0, -1)
    # Half angle of the cone (e.g., 30 degrees)
    cone_half_angle = np.pi / 6
    # Number of rays to generate
    num_rays = 100

    # Generate rays for the cone with the custom origin and cone parameters
    ray_origins, ray_directions = generate_rays_for_3d_conical_scan(
        origins=origin,
        directions=direction,
        half_angle=cone_half_angle,
        num_rays=num_rays,
        quaternions=rot
    )
    print('number of scan dots', ray_origins.shape)

    # Perform ray-mesh intersection
    locations, index_ray, index_tri = combined_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)

    # Filter points within a distance of 1.5
    relative_locations = locations - custom_origin
    distances = np.linalg.norm(relative_locations, axis=1)
    locations_within_range = locations[distances <= 2]

    # Visualize the results
    scene = trimesh.Scene()

    # Add the mesh to the scene
    scene.add_geometry(combined_mesh)

    # Add intersection points as a point cloud
    intersection_points = trimesh.PointCloud(locations_within_range, colors=[255, 0, 0])  # Red points
    scene.add_geometry(intersection_points)

    # Add rays as lines
    ray_visualize = trimesh.load_path(
        np.hstack((ray_origins, ray_origins + ray_directions * 0.2)).reshape(-1, 2, 3))
    scene.add_geometry(ray_visualize)

    # Show the scene
    scene.show()

if __name__ == "__main__":
    main()