import trimesh
import noise
import numpy as np
import open3d as o3d
import os
from collections.abc import Iterable

height = 6 # Height of the walls
# gap = 1.15  # Gap between the walls
gap = 1.05  # Gap between the walls

# width=20
width=10000+10
# width =1024+10  # Width of the walls
roughness_scale = 0.1 # Roughness intensity
resolution_xy = 1
resolution_z = 0.05
mesh_file_base_name  = "parallel_rough_walls"

width=100
mesh_file_base_name  = "parallel_rough_walls_200m"

# gap = 1.1
# gap = [1.0, 1.1, 1.1, 1.15, 1.0]  # Gap between the walls


gap = [0.98, 1.0, 1.02]  # Gap between the walls
width = 10000+10  # Width of the walls
roughness_scale = 0 # Roughness intensity
resolution_xy = width/50
resolution_z = height/10
mesh_file_base_name  = "parallel_smooth_walls_0.98_1.0_1.2m_long"

gap = [2,]
mesh_file_base_name  = "parallel_smooth_walls_2m_long"

# gap = 1.5  # Gap between the walls
# mesh_file_base_name = "tmp_debug"
import time
start_time = time.time()


# --- Create Vertices (Grid) ---

grid_size_x = int(width/resolution_xy) #width/resolution
grid_size_z = int(height/resolution_z) #height/resolution

x = np.linspace(-width / 2, width / 2, grid_size_x)
z = np.linspace(0, height, grid_size_z)
x_grid, z_grid, = np.meshgrid(x, z, indexing='xy')

import vnoise
noise = vnoise.Noise()
# if gap is iterable, use a loop to generate y_grid_1 and y_grid_2
if isinstance(gap, Iterable):
    # normalize the x grid to [0, 1] and use this as y_grid scaling
    y_grid = np.interp(
        x_grid,
        np.linspace(-width / 2, width / 2, len(gap)),
        gap
    )
else:
    y_grid = gap[0]

# y_grid_1 = np.zeros_like(x_grid) - gap/2
# y_grid_2 = np.zeros_like(x_grid) + gap/2

# for i in range(grid_size_z):
#     for j in range(grid_size_x):
#         y_grid_1[i, j] =  noise.pnoise3(x_grid[i, j],y_grid_1[i, j],z_grid[i, j], octaves=4, persistence=0.9, lacunarity=2.0) 
#         y_grid_2[i, j] =  noise.pnoise3(x_grid[i, j],y_grid_2[i, j],z_grid[i, j], octaves=4, persistence=0.9, lacunarity=2.0) 
#         # y_grid_1[i, j] =  roughness_scale*noise.pnoise2(x_grid[i, j],z_grid[i, j], octaves=4, persistence=0.9, lacunarity=2.0) 



y_grid_1 = -y_grid/2 - roughness_scale*noise.noise2(x_grid, z_grid, octaves=4, persistence=0.9, lacunarity=2.0, grid_mode=False, base=-1)
y_grid_2 =  y_grid/2 + roughness_scale*noise.noise2(x_grid, z_grid, octaves=4, persistence=0.9, lacunarity=2.0, grid_mode=False, base=1)
# print(x_grid.shape)


# y_grid_1 *=roughness_scale
# y_grid_2 *=roughness_scale
# + np.random.uniform(-0.25*resolution_xy, 0.25*resolution_xy, grid_size_x)
vertices_1 = np.vstack([x_grid.ravel(), y_grid_1.ravel(), z_grid.ravel()]).T
vertices_2 = np.vstack([x_grid.ravel(), y_grid_2.ravel(), z_grid.ravel()]).T

end_time_1 = time.time()
print(f"Time taken to generate the grid: {end_time_1 - start_time:.2f} seconds")

# --- Vectorized Face Creation ---
# Create 2D arrays for i and j corresponding to each cell's bottom-left corner
i_indices, j_indices = np.indices((grid_size_z - 1, grid_size_x - 1), dtype=np.int64)
# Calculate the flattened indices of the four corners for all cells simultaneously
v0 = i_indices * grid_size_x + j_indices
v1 = i_indices * grid_size_x + (j_indices + 1)
v2 = (i_indices + 1) * grid_size_x + j_indices
v3 = (i_indices + 1) * grid_size_x + (j_indices + 1)
# Reshape the indices to be 1D arrays
v0_flat = v0.ravel()
v1_flat = v1.ravel()
v2_flat = v2.ravel()
v3_flat = v3.ravel()
# Create the two sets of triangles using vectorized operations
# Triangle 1: [v0, v1, v3]
faces1 = np.stack([v0_flat, v1_flat, v3_flat], axis=1)
# Triangle 2: [v0, v3, v2]
faces2 = np.stack([v0_flat, v3_flat, v2_flat], axis=1)
# Combine the two sets of faces
faces_array = np.concatenate([faces1, faces2], axis=0)

# # --- Create Faces (Triangulate the Grid) ---
# faces = []
# # Iterate through the grid cells (excluding the last row and column)
# for i in range(grid_size_z - 1):  # Iterate through rows (z-direction)
#     for j in range(grid_size_x - 1):  # Iterate through columns (x-direction)
#         # Get indices of the four corners of the current grid cell
#         # Indexing follows the flattened order of vertices
#         v0 = i * grid_size_x + j         # Bottom-left corner
#         v1 = i * grid_size_x + (j + 1)   # Bottom-right corner
#         v2 = (i + 1) * grid_size_x + j     # Top-left corner
#         v3 = (i + 1) * grid_size_x + (j + 1) # Top-right corner
#         # Create two triangles for each quad cell
#         faces.append([v0, v1, v3])  # Triangle 1 (bottom-left, bottom-right, top-right)
#         faces.append([v0, v3, v2])  # Triangle 2 (bottom-left, top-right, top-left)
#         # # Alternative triangulation (splits the diagonal the other way)
#         # faces.append([v0, v1, v2])
#         # faces.append([v1, v3, v2])
# # Convert the list of faces to a numpy array
# faces_array = np.array(faces, dtype=np.int64) # Use int64 for potentially large meshes

# vertices_1[:, 1] = vertices_1[:, 1] - gap/2
# vertices_2[:, 1] = vertices_2[:, 1] + gap/2

# Identify the indices of vertices on the bottom edge (first row, i=0)
# These are the first grid_size_x vertices in the original flattened arrays
bottom_indices_mesh1 = np.arange(grid_size_x)
bottom_indices_mesh2 = np.arange(grid_size_x) # Indices relative to mesh2's vertices

# # Create faces for the strip connecting the two bottom edges
mesh2_vert_offset = len(vertices_1)
v1_curr = bottom_indices_mesh1[:-1]
v1_next = bottom_indices_mesh1[1:]
v2_curr = bottom_indices_mesh2[:-1] + mesh2_vert_offset
v2_next = bottom_indices_mesh2[1:] + mesh2_vert_offset
# Create the two sets of triangles using vectorized operations
# Triangle 1: [v1_curr, v2_curr, v1_next]
faces1 = np.stack([v1_curr, v2_curr, v1_next], axis=1)
# Triangle 2: [v1_next, v2_curr, v2_next]
faces2 = np.stack([v1_next, v2_curr, v2_next], axis=1)
# Combine the two sets of faces
strip_faces_array = np.concatenate([faces1, faces2], axis=0)
# Apply the winding order reversal
strip_faces_array = strip_faces_array[:, ::-1]


all_faces = np.vstack((faces_array[:, ::-1],faces_array+mesh2_vert_offset, strip_faces_array))
all_vertices = np.vstack((vertices_1, vertices_2))
# combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
# print num of vertices and faces in one line
print(f"Number of vertices: {len(all_vertices)}, Number of faces: {len(all_faces)}")


end_time_2 = time.time()
# print(f"Time taken to generate the mesh: {end_time_2 - end_time_1:.2f} seconds")
# exit(0)




# # --- Plot y_grid as a Heightmap ---
# plt.figure(figsize=(20, 3)) # Adjust figure size for better aspect ratio

# # Use imshow to display the 2D y_grid array
# # extent defines the coordinates of the image bounds: [left, right, bottom, top]
# # origin='lower' puts the (0,0) index at the bottom-left corner
# # aspect='auto' adjusts the aspect ratio to fit the figure
# im = plt.imshow(y_grid, cmap='terrain',
#                 extent=[x.min(), x.max(), z.min(), z.max()],
#                 origin='lower', aspect='auto')

# plt.colorbar(im, label='Y value (Height/Noise)') # Add a colorbar with label
# plt.xlabel("X coordinate")
# plt.ylabel("Z coordinate")
# plt.title("Heightmap of Generated Y Values (Noise Field)")
# plt.show()


def simplify_o3d_mesh(o3d_mesh:o3d.geometry.TriangleMesh, decimation:int,maximum_error=0.001):
    if decimation<=1:
        return o3d_mesh 
    # .remove_unreferenced_vertices()\
    # .remove_duplicated_vertices()\
    # .remove_duplicated_triangles()\
    # .remove_degenerate_triangles()\
    # .remove_non_manifold_edges()\
    # .merge_close_vertices(maximum_error)\
    # .simplify_vertex_clustering(maximum_error)\
    mesh_smp = o3d_mesh\
        .simplify_quadric_decimation(
            target_number_of_triangles=int(len(o3d_mesh.triangles)/decimation),
            maximum_error = maximum_error
            )
    mesh_smp.compute_vertex_normals()
    mesh_smp.compute_triangle_normals()
    print(mesh_smp)
    return mesh_smp

def vf_to_mesh(vertices, faces, return_type="o3d"):
    """Converts vertex and face arrays into a mesh object.
    """
    if return_type == "o3d":
        return o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )
    elif return_type == "trimesh":
        return trimesh.Trimesh(vertices=vertices, faces=faces)


# https://pyvista.github.io/fast-simplification/
import fast_simplification
t1 = time.time()
vertices_simp, faces_simp = fast_simplification.simplify(all_vertices, all_faces, target_reduction=0.9,verbose=True,agg=3)

# combined_mesh_o3d = vf_to_mesh(vertices_simp, faces_simp, return_type="o3d")
# # combined_mesh_o3d = vf_to_mesh(all_vertices, all_faces, return_type="o3d")
# simp_combined_mesh_o3d = simplify_o3d_mesh(combined_mesh_o3d, decimation=10,maximum_error=0.001)
# simp_combined_mesh = vf_to_mesh(simp_combined_mesh_o3d.vertices, simp_combined_mesh_o3d.triangles, return_type="trimesh")
# vertices_simp = simp_combined_mesh.vertices
# faces_simp = simp_combined_mesh.faces

t2 = time.time()
print(f"Time taken to simplify the mesh: {t2 - t1:.2f} seconds")
# print num of vertices and faces in one line
print(f"simplified Number of vertices: {len(vertices_simp)}, Number of faces: {len(faces_simp)}")

t1 = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.abspath(f"{current_dir}/../../assets/urdf/climbup/{mesh_file_base_name}.npz")
np.savez(save_path, vertices=vertices_simp.astype(np.float32), faces=faces_simp.astype(np.uint32),allow_pickle=False)
t2 = time.time()
print(f"Time taken to save the mesh: {t2 - t1:.2f} seconds")

# simp_combined_mesh = vf_to_mesh(vertices_simp,faces_simp, return_type="trimesh")
# # # simp_combined_mesh = combined_mesh
# # # simp_combined_mesh.show()
# # end_time_3 = time.time()
# # print(f"Time taken to simplify the mesh: {end_time_3 - end_time_2:.2f} seconds")
# # # --- Visualization ---
# # # Create a scene containing your mesh
# # scene = trimesh.Scene(simp_combined_mesh)
# # # Show the scene, disabling back-face culling
# # print("Showing scene with back-face culling DISABLED (both sides visible)...")
# # # Pass flags to disable culling.
# # # 'smooth=False' is often helpful to see the flat faces clearly.
# # # 'viewer='gl'' explicitly requests the standard OpenGL viewer.
# # # scene.show(viewer='gl', smooth=False, flags={'cull': False})
# # scene.show(viewer='gl', smooth=False)
# # Export as an OBJ file (optional, for Isaac Gym)
# # get the path to the current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# save_path = os.path.abspath(f"{current_dir}/../../assets/urdf/climbup/{mesh_file_base_name}.obj")
# simp_combined_mesh.export(save_path)