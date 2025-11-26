
import open3d as o3d
import numpy as np
from data_manager import load_las_and_visualize, point_cloud_sphere
import matplotlib.pyplot as plt

radius = 0.1  # Search radius (e.g., 10cm)
max_nn = 30   # Max number of nearest neighbors

def compute_tangent_plane_curvature(pcd):

    # Define the search parameters for local neighborhood
    # Adjust these based on the density and noise of your point cloud
    print("Estimating point normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    # Optional: Orient the normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)

    # Optional: Orient normals consistently (e.g., all pointing outward)
    #pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))    

    #3. Visualize Normals
    print("Visualizing point cloud with normals...")

    # Use the draw_geometries function and set point_show_normals to True
    o3d.visualization.draw_geometries([pcd],
                                    window_name="Point Cloud with Normals",
                                    # --- Key Parameter ---
                                    point_show_normals=True,
                                    # You can also control the line size and zoom for the initial view
                                    zoom=0.8,
                                    front=[0.4257, -0.2125, -0.8795],
                                    up=[-0.4795, -0.8751, -0.0875],
                                    lookat=[2.6172, 2.0475, 1.5323])



def compute_approximate_curvature(pcd, radius, max_nn):
    
    """
    Approximates curvature based on the eigenvalues of the local covariance matrix.
    Uses the ratio of the smallest eigenvalue (normal direction) to the sum.
    Curvature ApproximationWe can approximate the principal curvatures by examining the eigenvalues. 
    A common method, derived from fitting a Monge patch, relates the eigenvalues to the surface variation.We can use the ratio of the smallest eigenvalue ($\lambda_0$) to the sum of all eigenvalues ($\lambda_{sum} = \lambda_0 + \lambda_1 + \lambda_2$) as a proxy for the total curvature (or surface change) at that point.
    """
    points = np.asarray(pcd.points)
    curvatures = np.zeros(len(points))

    # Create a KDTree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(len(points)):
        # Find k nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], max_nn)
        
        # Get the coordinates of the neighbors
        neighbors = points[idx, :]
        
        # Center the neighbors
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute the covariance matrix
        cov_matrix = np.cov(neighbors_centered.T)
        
        # Get the eigenvalues
        eigenvalues = np.linalg.eigh(cov_matrix)[0]
        
        # Sort eigenvalues (ascending order: lambda_0 < lambda_1 < lambda_2)
        eigenvalues.sort()
        
        lambda_0 = eigenvalues[0]
        lambda_sum = np.sum(eigenvalues)
        
        # Curvature proxy: measures how much the cloud spreads along the normal direction
        if lambda_sum > 1e-8:
            # Planarity/Surface Variation proxy: Smaller values indicate a more planar, less curved surface
            curvatures[i] = lambda_0 / lambda_sum  
        else:
            curvatures[i] = 0.0

    return curvatures

# --- Execution ---
def compute_and_show_curvature(pcd):

    # 1. Load or Generate Point Cloud
    # pcd = o3d.io.read_point_cloud("path/to/your/point_cloud.pcd")
    #pcd = point_cloud_sphere()  # Using sphere point cloud for demonstration

    # 2. Compute Tangent Plane Normals
    #compute_tangent_plane_curvature(pcd)

    # 3. Compute Approximate Curvature
    # You must call this *after* normal estimation if you want to reuse the neighborhood size
    curvatures = compute_approximate_curvature(pcd, radius, max_nn)

    print(f"Computed {len(curvatures)} curvature values.")
    print(f"Min Curvature: {np.min(curvatures):.4f}")
    print(f"Max Curvature: {np.max(curvatures):.4f}")

    # 4. Color the Point Cloud by Curvature
    # Normalize the curvature values to the range [0, 1] for color mapping
    curvatures_norm = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))

    # Use a colormap (e.g., 'plasma' or 'jet') to visualize curvature

    cmap = plt.get_cmap("plasma")
    colors = cmap(curvatures_norm)[:, :3] # Get RGB without alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Colored by Curvature")

#%% Test Class
class TestCurvature:
    def test_compute_and_show_curvature(self):
        pcd = point_cloud_sphere()
        compute_and_show_curvature(pcd)
#%%
if __name__ == '__main__':
    # 1. Working
    tst = TestCurvature()
    tst.test_compute_and_show_curvature()
