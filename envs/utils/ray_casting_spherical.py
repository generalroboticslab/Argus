import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

# Function to generate ray origins and directions for a spherical scan
def generate_rays_for_spherical_scan(origin=(0, 0, 0), radius=1, resolution=50):
    """
    Generate ray origins and directions for a spherical scan with a custom origin.

    Parameters:
        origin (tuple): The origin (x, y, z) from which rays originate.
        radius (float): Radius of the sphere from which rays originate.
        resolution (int): Number of rays to generate.

    Returns:
        tuple: Arrays of ray origins and ray directions.
    """
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution // 2)
    
    print('phi', phi.shape, 'theta', theta.shape)
    phi, theta = np.meshgrid(phi, theta)
    print('phi', phi.shape, 'theta', theta.shape)

    # Calculate the ray directions based on spherical coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Create the ray origins by adding the provided origin to the computed directions
    origins = np.full((len(x.ravel()), 3), origin)  # Rays originate from the custom origin
    directions = np.vstack((x.ravel(), y.ravel(), z.ravel())).T

    return origins, directions

# Main function to perform the spherical scan using ray intersections
def main():
    # Create a sample mesh (e.g., a sphere)
    radius = 0.55
    resolution = 15
    mesh = trimesh.load("../assets/urdf/cube/cube_multicolor.obj", force='mesh', skip_materials=True)

    plane = trimesh.primitives.Box(extents=[20, 20, 0.1], transform=trimesh.transformations.translation_matrix([0, 0, -0.1/2]))
    mesh = trimesh.primitives.Box(extents=[3 ,1 ,0.3], transform=trimesh.transformations.translation_matrix([0, 2, 1.3]))

    combined_mesh = trimesh.util.concatenate([mesh, plane])

    # Custom origin for ray generation (example: (1, 1, 1))
    custom_origin = (1, 2, 0.5)

    # Generate rays for the scan with the custom origin
    ray_origins, ray_directions = generate_rays_for_spherical_scan(origin=custom_origin, radius=radius, resolution=resolution)

    print('number of scan dots', ray_origins.shape)

    # Perform ray-mesh intersection
    locations, index_ray, index_tri = combined_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions,multiple_hits=False)
    
    # locations = locations - custom_origin  # Translate the intersection points to the origin
    ray_visualize = trimesh.load_path(
        np.hstack((ray_origins, ray_origins + ray_directions * 1.0)).reshape(-1, 2, 3))

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
    scene.add_geometry(ray_visualize)

    # Show the scene
    scene.show()

if __name__ == "__main__":
    main()
