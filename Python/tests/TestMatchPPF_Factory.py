
# Test matching on real data

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

import laspy
import open3d as o3d
import cv2 as cv
import numpy as np
import time

#%% Data load
def data_load(fname):
    las = laspy.read(fname)
    las.header.scales # dimensions
    point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
    return point_data




#%% Show
def build_show(point_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)
    o3d.visualization.draw_geometries([pcd])
    return pcd

def compute_keypoints(pcd):
    tic = time.time()
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    #mesh.compute_vertex_normals()
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints.paint_uniform_color([1.0, 0.75, 0.0])
    o3d.visualization.draw_geometries([keypoints, pcd])
    return keypoints

def get_icp_transform(source, target, source_indices, target_indices):
    corr = np.zeros((len(source_indices), 2))
    corr[:, 0] = source_indices
    corr[:, 1] = target_indices

    # Estimate rough transformation using correspondences
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # Point-to-point ICP for refinement
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return reg_p2p.transformation


#%% Main
fname_pump  = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\pump.laz'
fname_press = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\pressure.laz'
fname_valve = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\valve.laz'
fname_farm  = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\farm2M.laz'

data_pump   = data_load(fname_pump)
#data_press  = data_load(fname_press)
#data_valve  = data_load(fname_valve)
data_farm   = data_load(fname_farm)

pcd_pump    = build_show(data_pump)
#pcd_press   = build_show(data_press)
#pcd_valave  = build_show(data_valve)
pcd_farm    = build_show(data_farm)

key_pump    = compute_keypoints(pcd_pump)
#key_press   = compute_keypoints(data_press)
#key_valave  = compute_keypoints(data_valve)
key_farm    = compute_keypoints(pcd_farm)