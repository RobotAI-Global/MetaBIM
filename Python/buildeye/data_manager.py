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
    pcd         = o3d.geometry.PointCloud()
    # Set the points/vertices of the PointCloud
    pcd.points  = o3d.utility.Vector3dVector(coords)

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
    #pcd.compute_vertex_normals()
    pcd.paint_uniform_color([0.9, 0.6, 0.1]) # Orange color

    # 2. Sample the mesh surface to get a Point Cloud
    #pcd = pcd.sample_points_uniformly(number_of_points=20000)    
    # 3. Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Random Cloud")      
    return pcd

# --- Sphere Generation ---
def point_cloud_sphere():
    print("Generating Sphere Point Cloud...")

    # 1. Create a Sphere Mesh
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere_mesh.compute_vertex_normals()
    sphere_mesh.paint_uniform_color([0.1, 0.7, 0.9]) # Blue color

    # 2. Sample the mesh surface to get a Point Cloud
    pcd         = sphere_mesh.sample_points_uniformly(number_of_points=2000)    

    # highlight specific points
    indices     = [1,501,1001, 1501]  # 

    # 3. Initialize all colors to gray
    colors      = np.full((len(pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red  
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue    

    # 5. Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)     

    # 3. Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Sphere Point Cloud")

    return pcd
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

    indices         = [1,501,1001,1501]  # 

    # 3. Initialize all colors to gray
    colors          = np.full((len(torus_pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue    

    # 5. Assign colors to point cloud
    torus_pcd.colors = o3d.utility.Vector3dVector(colors)         

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
    pcd         = cylinder_mesh.sample_points_uniformly(number_of_points=2000)

    # highlight specific points
    indices     = [1,501,1001, 1501]  # 

    # 3. Initialize all colors to gray
    colors      = np.full((len(pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red  
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue    

    # 5. Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)       

    # 6. Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Cylinder Point Cloud")

    return pcd

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

    return pcd

def point_cloud_box():
    # 1. Create a Box Mesh
    print("Generating Box Point Cloud...")
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    box_mesh.compute_vertex_normals()
    box_mesh.paint_uniform_color([0.2, 0.8, 0.2]) # Green color

    # 2. Sample the mesh surface to get a Point Cloud
    box_pcd = box_mesh.sample_points_uniformly(number_of_points=1200)

    # 3. Visualize
    o3d.visualization.draw_geometries([box_pcd], window_name="Box Point Cloud")

    return box_pcd

def point_cloud_plane_random(size=1.0, num_points=2000, switch_yz = False, switch_xz = False):
    """
    Create a planar point cloud with random points.
    
    Parameters:
        size (float): Side length of the square plane
        num_points (int): Number of random points
    
    Returns:
        open3d.geometry.PointCloud
    """
    x = np.random.uniform(0, size, num_points)
    y = np.random.uniform(0, size, num_points)
    z = np.zeros(num_points)
    if switch_yz:
        y,z = z,y
    if switch_xz:
        x,z = z,x        
    
    points = np.column_stack((x, y, z))
    points = points + np.random.normal(scale=0.001, size=points.shape)  # Add slight noise
    
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(points)

    # Optional: color
    plane_pcd.colors = o3d.utility.Vector3dVector([[0.5, 0.9, 0.5]] * len(plane_pcd.points))

    # 2. Identify indices to color (e.g., x > 0.5)
    #points_np = np.asarray(plane_pcd.points)
    #indices = np.where(points_np[:, 0] > 0.5)[0]  # All points with x > 0.5

    # 3. Initialize all colors to gray
    colors          = np.full((len(plane_pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # highlight specific points
    indices         = [1,501,1001,1501]  # 

    # 3. Initialize all colors to gray
    colors          = np.full((len(plane_pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue     

    # 5. Assign colors to point cloud
    plane_pcd.colors = o3d.utility.Vector3dVector(colors)    

    o3d.visualization.draw_geometries([plane_pcd])    
    return plane_pcd

def point_cloud_corner(size=1.0, num_points=2000):
    """
    Create a corner - 2 plane intersection  point cloud with random points.
    
    Parameters:
        size (float): Side length of the square plane
        num_points (int): Number of random points
    
    Returns:
        open3d.geometry.PointCloud
    """
    # plane 1
    x = np.random.uniform(0, size, num_points//2)
    y = np.random.uniform(0, size, num_points//2)
    z = np.zeros(num_points//2)
    points1 = np.column_stack((x, y, z))

    # plane 2 - orthogonal
    x = np.random.uniform(0, size, num_points//2)
    z = np.random.uniform(0, size, num_points//2)
    y = np.zeros(num_points//2)
    points2 = np.column_stack((x, y, z))
    
    points      = np.vstack((points1, points2))
    points      = points + np.random.normal(scale=0.0, size=points.shape)  # Add slight noise
    
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)

    # Optional: color
    #pcd.colors = o3d.utility.Vector3dVector([[0.5, 0.9, 0.5]] * len(pcd.points))

    # highlight specific points
    indices         = [1,501,1001,1501]  # 

    # 3. Initialize all colors to gray
    colors          = np.full((len(pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue     

    # 5. Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)     

    o3d.visualization.draw_geometries([pcd],window_name="Corner Point Cloud")    
    return pcd

def point_cloud_curved_plane(size=1.0, num_points=2000):
    """
    Create a curved plane - like cylinder  point cloud with random points.
    
    Parameters:
        size (float): Side length of the square plane
        num_points (int): Number of random points
    
    Returns:
        open3d.geometry.PointCloud
    """
    # plane 1
    x = np.random.uniform(0, size, num_points)
    y = np.random.uniform(0, size, num_points)
    #z = np.cos(np.arctan((x - size//2)/(size))) * size
    z = np.sqrt(size**2 - (x - size//2)**2)
    points = np.column_stack((x, y, z))

    points      = points + np.random.normal(scale=0.0, size=points.shape)  # Add slight noise
    
    pcd         = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)

    # Optional: color
    #pcd.colors = o3d.utility.Vector3dVector([[0.5, 0.9, 0.5]] * len(pcd.points))

    # highlight specific points
    indices         = [1,501,1001,1501]  # 

    # 3. Initialize all colors to gray
    colors          = np.full((len(pcd.points), 3), 0.7)  # Light gray (0.7, 0.7, 0.7)

    # 4. Set specific points to red
    colors[indices]    = [1.0, 0.0, 0.0]  # Red
    colors[indices[0]] = [0.0, 1.0, 0.0]  # Green 
    colors[indices[1]] = [0.0, 0.0, 1.0]  # Blue     

    # 5. Assign colors to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)     

    o3d.visualization.draw_geometries([pcd],window_name="Corner Point Cloud")    
    return pcd

#%%
if __name__ == '__main__':
    # 1. Working
    file_path = "Data/onyx.las" 
    #load_las_and_visualize(file_path)

    # 2. Sphere Point Cloud -ok
    #point_cloud_sphere()

    # 3. Torus Point Cloud -ok
    #point_cloud_torus()

    # 4. Random Point Cloud - ok
    #point_cloud_random() 

    # 5. Cylinder Point Cloud - ok
    #point_cloud_cylinder()
    # 
    # 6. Bunny Point Cloud - ok
    #    #point_cloud_bunny()
    
    # 7. Box Point Cloud - ok
    #point_cloud_box()  

    # 8. Plane Random Point Cloud - ok
    #point_cloud_plane_random(size=1.0, num_points=2000)

    # Corner Point Cloud - ok
    #point_cloud_corner(size=1.0, num_points=1000)

    # curved plane
    point_cloud_curved_plane()