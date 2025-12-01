'''
BuildEye data loader and show

'''
import open3d as o3d
import numpy as np
import laspy
import os

def load_las_and_visualize(file_path=None):
    """
    Loads a LAS file using laspy, converts it to an Open3D PointCloud, 
    and displays it in a 3D viewer.
    """
    if file_path is None:
        file_path = "Data/onyx.las" 

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading LAS file: {file_path}")
    
    # 1. Read the LAS file using laspy
    try:
        with laspy.open(file_path) as fh:
            las_data = fh.read()
    except Exception as e:
        print(f"Error reading LAS file with laspy: {e}")
        return

    # Extract coordinates (X, Y, Z) and stack them into an Nx3 NumPy array
    # Laspy handles scale and offset automatically when accessing .x, .y, .z
    coords = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
    print(f"Read {len(coords)} points.")

    # 2. Convert NumPy array to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    # Set the points/vertices of the PointCloud
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Optional: Handle point colors if intensity data is available
    if "intensity" in las_data.point_format.dimension_names:
        print("Mapping intensity data to colors.")
        # Normalize intensity for color mapping (0 to 1 range)
        intensity = las_data.intensity.astype(np.float64)
        # Add a small epsilon to the denominator to prevent division by zero
        intensity_norm = (intensity - intensity.min()) / (np.ptp(intensity) + 1e-8)
        
        # Use a colormap to convert scalar intensity to RGB colors
        # You may need to install matplotlib for this: `pip install matplotlib`
        try:
            import matplotlib.pyplot as plt
            # 'viridis' is a common colormap. [:, :3] selects the RGB channels, ignoring alpha
            colors = plt.get_cmap("viridis")(intensity_norm)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        except ImportError:
            print("Warning: matplotlib not installed. Points will be default color.")


    # 3. Visualize the PointCloud
    print("Visualizing point cloud. Press 'H' in the viewer for controls.")
    o3d.visualization.draw_geometries([pcd])

    return pcd


# --- Sphere Generation ---
def point_cloud_random():
    print("Generating Random Point Cloud...")
    # Assume 'pcd' is your loaded Open3D PointCloud object
    # For demonstration, let's create a simple sphere point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3)) 
    # If you have a real point cloud, load it here
    #pcd = o3d.io.read_point_cloud("path/to/your/point_cloud.pcd")
    pcd.compute_vertex_normals()
    pcd.paint_uniform_color([0.9, 0.6, 0.1]) # Orange color

    # 2. Sample the mesh surface to get a Point Cloud
    las_pcd = pcd.sample_points_uniformly(number_of_points=20000)    
    return las_pcd

# --- Sphere Generation ---
def point_cloud_sphere():
    print("Generating Sphere Point Cloud...")

    # 1. Create a Sphere Mesh
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere_mesh.compute_vertex_normals()
    sphere_mesh.paint_uniform_color([0.1, 0.7, 0.9]) # Blue color

    # 2. Sample the mesh surface to get a Point Cloud
    # The 'number_of_points' controls the density of the point cloud
    sphere_pcd = sphere_mesh.sample_points_uniformly(number_of_points=1000)

    # 3. Visualize
    o3d.visualization.draw_geometries([sphere_pcd], window_name="Sphere Point Cloud")

    return sphere_pcd
#
def point_cloud_torus():
    # --- Torus Generation ---
    print("Generating Torus Point Cloud...")

    # 1. Create a Torus Mesh
    # tube_radius: the radius of the tube itself
    # ring_radius: the radius from the center to the tube center
    torus_mesh = o3d.geometry.TriangleMesh.create_torus()
    #     tube_radius=0.2, 
    #     ring_radius=0.8
    # )
    torus_mesh.compute_vertex_normals()
    torus_mesh.paint_uniform_color([0.9, 0.6, 0.1]) # Orange color

    # 2. Sample the mesh surface to get a Point Cloud
    torus_pcd = torus_mesh.sample_points_uniformly(number_of_points=2000)

    # 3. Visualize
    o3d.visualization.draw_geometries([torus_pcd], window_name="Torus Point Cloud")

    return torus_pcd
#

def point_cloud_cylinder():
    # --- Cylinder Generation ---
    print("Generating Cylinder Point Cloud...")

    # 1. Create a Cylinder Mesh
    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.5, 
        height=2.0
    )
    cylinder_mesh.compute_vertex_normals()
    cylinder_mesh.paint_uniform_color([0.6, 0.2, 0.8]) # Purple color

    # 2. Sample the mesh surface to get a Point Cloud
    cylinder_pcd = cylinder_mesh.sample_points_uniformly(number_of_points=1500)

    # 3. Visualize
    o3d.visualization.draw_geometries([cylinder_pcd], window_name="Cylinder Point Cloud")

def point_cloud_bunny():
    # 1. Load a Point Cloud (or create one)
    print("Loading point cloud...")
    # We'll use the built-in Stanford Bunny mesh for a good example surface
    bunny_mesh = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny_mesh.path)

    # Convert the mesh to a point cloud by sampling the surface
    pcd = mesh.sample_points_uniformly(number_of_points=5000)

    # 3. Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Bunny Point Cloud")    

#%%
if __name__ == '__main__':
    # 1. Working
    file_path = "Data/onyx.las" 
    load_las_and_visualize(file_path)

    # 2. Sphere Point Cloud -ok
    #point_cloud_sphere()

    # 3. Torus Point Cloud -ok
    #point_cloud_torus()

    # 4.
    