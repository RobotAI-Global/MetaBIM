# Test Open3D

# # Conda:
#     conda create --prefix D:\Uri\Programs\miniconda3\envs\metabim python=3.10

# # Env on old home PC : 
#     (metabim) D:\Uri\Hamlet\Customers\MetaBIM\

# Installs: 
#     pip install open3d
#     pip install laspy[lazrs,laszip]
#     pip install pye57

#     pip install -U scikit-learn
#     pip install matplotlib
#     pip install opencv-contrib-python

import open3d as o3d
#import cv2 as cv
import numpy as np
import time

#%% Demo Pose Graph Optimization
def demo_pose():
    dataset = o3d.data.DemoPoseGraphOptimization()
    pose_graph_fragment = o3d.io.read_pose_graph(dataset.pose_graph_fragment_path)
    pose_graph_global = o3d.io.read_pose_graph(dataset.pose_graph_global_path)

#%% Voxel-Block-Grid
# examples/python/t_reconstruction_system/integrate.py
""" if config.integrate_color:
    vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=('tsdf', 'weight', 'color'),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=3.0 / 512,
                block_resolution=16,
                block_count=50000,
                device=device)
else:
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
                                                attr_dtypes=(o3c.float32,
                                                             o3c.float32),
                                                attr_channels=((1), (1)),
                                                voxel_size=3.0 / 512,
                                                block_resolution=16,
                                                block_count=50000,
                                                device=device) """
 
 #%% KD
def kd_tree_feature_match_1():
    print("Load two aligned point clouds.")
    demo_data = o3d.data.DemoFeatureMatchingPointClouds()
    pcd0 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd0, pcd1])
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.fpfh_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.fpfh_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")

def kd_tree_feature_match_2():
    print("Load their L32D feature and evaluate.")
    demo_data = o3d.data.DemoFeatureMatchingPointClouds()
    pcd0 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
    pcd1 = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])

    
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = o3d.io.read_feature(demo_data.l32d_feature_paths[0])
    feature1 = o3d.io.read_feature(demo_data.l32d_feature_paths[1])

    fpfh_tree = o3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    o3d.visualization.draw_geometries([pcd0])
    print("")                                               
 
def radius_search():
    print("Loading pointcloud ...")
    sample_pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the neighbors of 50000th point with distance less than 0.2, and painting them green ..."
    )
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50000], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])


def knn_search():
    print("Loading pointcloud ...")
    sample_pcd = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd.path)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print(
        "Find the 2000 nearest neighbors of 50000th point, and painting them red ..."
    )
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[50000], 2000)
    np.asarray(pcd.colors)[idx[1:], :] = [1, 0, 0]

    print("Displaying the final point cloud ...\n")
    o3d.visualization.draw([pcd])

                
#%% Oct Tree
def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # We only want to process nodes / spatial regions with enough points.
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # Early stopping: if True, traversal of children of the current node will be skipped.
    return early_stop

def octtree_show():
    N = 2000
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(N)
    # Fit to unit cube.
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,
                                                              size=(N, 3)))

    octree = o3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    print('Displaying input octree ...')
    o3d.visualization.draw([octree])
    print('Traversing octree ...')
    octree.traverse(f_traverse) 
    
#%% poinyt cloud fartherst sampling
def farthest_sampling():
    # Load bunny data.
    bunny = o3d.data.BunnyMesh()
    pcd = o3d.io.read_point_cloud(bunny.path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Get 1000 samples from original point cloud and paint to green.
    pcd_down = pcd.farthest_point_down_sample(1000)
    pcd_down.paint_uniform_color([1, 0, 0])
    pcd_down.translate((0.01, 0.01, 0.01))
    #o3d.visualization.draw_geometries([pcd, pcd_down])
    o3d.visualization.draw([pcd, pcd_down])
    #o3d.visualization.draw([pcd_down])
                  
#%% normls
def compute_normals():
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(5000)
    # Invalidate existing normals.
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    print("Displaying input pointcloud ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    pcd.estimate_normals()
    print("Displaying pointcloud with normals ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    print("Printing the normal vectors ...")
    print(np.asarray(pcd.normals))
                                                
if __name__ == "__main__":

    #kd_tree_feature_match_1()
    #knn_search()
    radius_search()
    #octtree_show()
    #farthest_sampling()
    #compute_normals()