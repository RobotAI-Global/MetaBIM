
# Test matching using example
# http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

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
import copy

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
    #o3d.visualization.draw_geometries([pcd])
    return pcd


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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



#%% Main
fname_pump  = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\pump.laz'
fname_press = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\pressure.laz'
fname_valve = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\valve.laz'
fname_farm  = r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\farm2M.laz'

data_pump   = data_load(fname_pump)
data_farm   = data_load(fname_farm)

pcd_pump    = build_show(data_pump)
pcd_farm    = build_show(data_farm)

voxel_size  = 10
pump_down, pump_fpfh = preprocess_point_cloud(pcd_pump, voxel_size)
farm_down, farm_fpfh = preprocess_point_cloud(pcd_farm, voxel_size)


result_ransac = execute_global_registration(pump_down, farm_down,
                                            pump_fpfh, farm_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(pump_down, farm_down, result_ransac.transformation)

result_icp = refine_registration(pcd_pump, pcd_farm, pump_fpfh, farm_fpfh, voxel_size)
print(result_icp)
draw_registration_result(pcd_pump, pcd_farm, result_icp.transformation)