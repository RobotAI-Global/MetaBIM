'''
Docstring for Python.buildeye.compute_curvature
Compute and visualize curvature of point clouds using Open3D.
'''


import open3d as o3d
import numpy as np
from data_manager import load_las_and_visualize, point_cloud_sphere, point_cloud_torus, point_cloud_cylinder, point_cloud_box
from data_manager import point_cloud_plane_random, point_cloud_corner, point_cloud_curved_plane
import matplotlib.pyplot as plt

radius = 0.1  # Search radius (e.g., 10cm)
max_nn = 30   # Max number of nearest neighbors

def compute_tangent_plane_curvature(pcd):

    # get dimnsions of the point cloud
    obb = pcd.get_oriented_bounding_box()
    oriented_extent = obb.extent  # [length along object's local x, y, z]
    print("Oriented dimensions:", oriented_extent)

    # Define the search parameters for local neighborhood
    # Adjust these based on the density and noise of your point cloud
    print("Estimating point normals...")
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    pcd.estimate_normals(search_param=search_param)

    # Optional: Orient the normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)

    # Optional: Orient normals consistently (e.g., all pointing outward)
    #pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))  
    # 
    # Convert the mesh to a point cloud by sampling the surface
    #pcd         = pcd.sample_points_uniformly(number_of_points=10000)  

    #3. Visualize Normals
    print("Visualizing point cloud with normals...")

    # # Use the draw_geometries function and set point_show_normals to True
    # o3d.visualization.draw_geometries([pcd],
    #                                 window_name="Point Cloud with Normals",
    #                                 # --- Key Parameter ---
    #                                 point_show_normals=True,
    #                                 # You can also control the line size and zoom for the initial view
    #                                 zoom=0.8,
    #                                 front=[0.4257, -0.2125, -0.8795],
    #                                 up=[-0.4795, -0.8751, -0.0875],
    #                                 lookat=[2.6172, 2.0475, 1.5323])
    
    # Visualize the point cloud with normals
    o3d.visualization.draw_geometries([pcd, obb], point_show_normal=True)
    
    return pcd

# --- Simple neighborhood structure analysis ---

def compute_approximate_curvature(pcd, radius, max_nn):
    """
    Approximates curvature based on the eigenvalues of the local covariance matrix.
    Uses the ratio of the smallest eigenvalue (normal direction) to the sum.
    Curvature ApproximationWe can approximate the principal curvatures by examining the eigenvalues. 
    A common method, derived from fitting a Monge patch, relates the eigenvalues to the surface variation.We can use the ratio of the smallest eigenvalue ($\lambda_0$) to the sum of all eigenvalues ($\lambda_{sum} = \lambda_0 + \lambda_1 + \lambda_2$) as a proxy for the total curvature (or surface change) at that point.
    """
    points = np.asarray(pcd.points)
    curvatures = np.zeros((len(points),3))

    # Create a KDTree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # debug
    #test_index = [1,501] # corner
    test_index = [1,501,1001,1501]  # cylinder, sphere

    for i in range(len(points)):

        # Find k nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], max_nn)
        
        # Get the coordinates of the neighbors
        neighbors = points[idx, :]
        
        # Center the neighbors
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute the covariance matrix u,s,v = np.linalg.svd(neighbors_centered)
        cov_matrix = np.cov(neighbors_centered.T)
        
        # Get the eigenvalues
        eigenvalues = np.linalg.eigh(cov_matrix)[0]

        # Sort eigenvalues (ascending order: lambda_0 < lambda_1 < lambda_2)
        #eigenvalues.sort()        

        # alternative way s is from max to min
        u,s,v       = np.linalg.svd(neighbors_centered)
        #eigenvalues = s
        curvatures[i,:] = s 

        if i in test_index:
            good_dir = np.dot(np.cross(v[0,:],v[1,:]),v[2,:])
            norm_dir = np.dot(pcd.normals[i],v[2,:])
            print(f'{i} ---- ')
            print(f'{good_dir:.1f} : {s} ')
            print(f'{v}')
            print(f'{norm_dir:.1f} : {pcd.normals[i]}')
            
            pass        
        
    return curvatures


def compute_and_show_curvature(pcd):

    # 1. Load or Generate Point Cloud
    # pcd = o3d.io.read_point_cloud("path/to/your/point_cloud.pcd")
    #pcd = point_cloud_sphere()  # Using sphere point cloud for demonstration

    # 2. Compute Tangent Plane Normals
    pcd = compute_tangent_plane_curvature(pcd)

    # 3. Compute Approximate Curvature
    # You must call this *after* normal estimation if you want to reuse the neighborhood size
    curvatures = compute_approximate_curvature(pcd, radius, max_nn)

    # actual approximation
    curvatures_approx = curvatures[:,2]/curvatures.sum(axis=1)    

    print(f"Computed {len(curvatures_approx)} curvature values.")
    print(f"Min Curvature: {np.min(curvatures_approx):.4f}")
    print(f"Max Curvature: {np.max(curvatures_approx):.4f}")

    # 4. Color the Point Cloud by Curvature
    # Normalize the curvature values to the range [0, 1] for color mapping
    curvatures_norm = (curvatures_approx - np.min(curvatures_approx)) / (np.max(curvatures_approx) - np.min(curvatures_approx))

    # Use a colormap (e.g., 'plasma' or 'jet') to visualize curvature

    cmap        = plt.get_cmap("plasma")
    colors      = cmap(curvatures_norm)[:, :3] # Get RGB without alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Colored by Curvature")

# --- Fit neighborhood quadratic ---

def compute_quadratic_curvature(pcd, radius, max_nn):
    """
    Approximates curvature based on the quadratic fit.
    Uses Goldfeather method to estimate principal values.
    z(x,y) = ax^2 + bxy + xy^2. 
    Deriving Weingarten matrix W = [[a,b],[b,c]] with eigenvalues and eigenvectors as principals curvature.
    """
    points    = np.asarray(pcd.points)
    curvatures = np.zeros((len(points),6))

    # Create a KDTree for efficient nearest neighbor search
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # debug
    #test_index = [1,501] # corner
    test_index = [1,501,1001,1501]  # cylinder, sphere

    for i in range(len(points)):

        # Find k nearest neighbors
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], max_nn)
        
        # Get the coordinates of the neighbors
        neighbors = points[idx, :]
        
        # Center the neighbors
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute the vandermond matrix
        xx_yy_zz         = neighbors_centered**2
        xy_yz            = neighbors_centered[:,:2]*neighbors_centered[:,1:]
        xz               = (neighbors_centered[:,0]*neighbors_centered[:,2]).reshape((-1,1))
        vander_matrix    = np.hstack((xx_yy_zz,xy_yz,xz))
        
        # alternative way s is from max to min
        u,s,v           = np.linalg.svd(vander_matrix)
        #eigenvalues = s
        a               = v[-1,:] 
        curvatures[i,:] = v[-1,:] 

        # read and construct and solve W
        #abcd            = a/a[2]  # make z2 coeff 1
        #W               = np.array([[abcd[0],abcd[3]],[abcd[3],abcd[1]]])

        # read and construct and solve W
        W               = np.array([[a[0],a[3],a[5]],[a[3],a[1],a[4]],[a[1],a[4],a[2]]])        

        # compute engevalues and vectors
        es, ev          = np.linalg.eig(W,)

        if i in test_index:
            good_dir = np.dot(np.cross(ev[:,0],ev[:,1]),ev[:,2])
            norm_dir = np.dot(pcd.normals[i],ev[:,2])
            print(f'{i} ---- ')
            print(f'{good_dir:.1f} : {es} ')
            print(f'{ev}')
            print(f'{norm_dir:.1f} : {pcd.normals[i]}')
            
            pass        
        

    return curvatures

def compute_and_show_quadratic(pcd):

    # 1. Load or Generate Point Cloud
    # pcd = o3d.io.read_point_cloud("path/to/your/point_cloud.pcd")
    #pcd = point_cloud_sphere()  # Using sphere point cloud for demonstration

    # 2. Compute Tangent Plane Normals
    pcd        = compute_tangent_plane_curvature(pcd)

    # 3. Compute Approximate Curvature
    # You must call this *after* normal estimation if you want to reuse the neighborhood size
    curvatures = compute_quadratic_curvature(pcd, radius, max_nn)

    # actual approximation
    curvatures_approx = curvatures[:,2]/curvatures.sum(axis=1)    

    print(f"Computed {len(curvatures_approx)} curvature values.")
    print(f"Min Curvature: {np.min(curvatures_approx):.4f}")
    print(f"Max Curvature: {np.max(curvatures_approx):.4f}")

    # 4. Color the Point Cloud by Curvature
    # Normalize the curvature values to the range [0, 1] for color mapping
    curvatures_norm = (curvatures_approx - np.min(curvatures_approx)) / (np.max(curvatures_approx) - np.min(curvatures_approx))


    # Use a colormap (e.g., 'plasma' or 'jet') to visualize curvature

    cmap        = plt.get_cmap("plasma")
    colors      = cmap(curvatures_norm)[:, :3] # Get RGB without alpha

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Colored by Curvature")


#%% Test Class
class TestCurvature:
    def test_compute_and_show_curvature(self):
        pcd = point_cloud_sphere()
        #pcd = point_cloud_torus()
        #pcd = point_cloud_cylinder()
        #pcd = point_cloud_box()
        #pcd = point_cloud_plane_random(size=1.0, num_points=2000, switch_yz = False, switch_xz = True)
        #pcd = point_cloud_corner(size=1.0, num_points=2000)
        #pcd = point_cloud_curved_plane()
        compute_and_show_curvature(pcd)

    def test_las_file_compute_and_show_curvature(self):
        #pcd = point_cloud_sphere()
        pcd = load_las_and_visualize()
        compute_and_show_curvature(pcd)    

    def test_compute_and_show_quadratic(self):
        #pcd = point_cloud_sphere()
        #pcd = point_cloud_torus()
        #pcd = point_cloud_cylinder()
        #pcd = point_cloud_box()
        #pcd = point_cloud_plane_random(size=1.0, num_points=2000, switch_yz = False, switch_xz = True)
        #pcd = point_cloud_corner(size=1.0, num_points=2000)
        pcd = point_cloud_curved_plane()
        compute_and_show_quadratic(pcd)            
#%%
if __name__ == '__main__':
    # 1. Working
    tst = TestCurvature()
    #tst.test_compute_and_show_curvature() # ok
    tst.test_las_file_compute_and_show_curvature()
    #tst.test_compute_and_show_quadratic()
