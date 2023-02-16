

# # Show how to work with data

# # Conda:
#     conda create --prefix D:\Uri\Programs\miniconda3\envs\metabim python=3.10

# # Env on old home PC : 
#     (metabim) D:\Uri\Hamlet\Customers\MetaBIM\

# Installs: 
#     pip install open3d
#     pip install laspy[lazrs,laszip]
#     pip install pye57

#     pip install -U scikit-learn

#%%

import laspy
import open3d as o3d
import numpy as np

#%%
#las = laspy.read(r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\pump.laz')
las = laspy.read(r'D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\farm2M.laz')
las

las.X
las.intensity
las.gps_time

list(las.point_format.dimension_names)
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
buildings.points = las.points[las.classification == 6]

#%% show
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([pcd])

#%% downsample
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%%
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)


print("Print a normal vector of the 0th point")
print(downpcd.normals[0])

print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])

#%% core
print("Load a polygon volume and use it to crop the original point cloud")
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

#%%

# pcd.points = o3d.utility.Vector3dVector(point_data)
# pcd.colors = o3d.utility.Vector3dVector(colors/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([pcd])

#%%

# voxel_grid = o3d.geometry.VoxelGrid
# create_from_point_cloud(geom,voxel_size=0.40)
# o3d.visualization.draw_geometries([voxel_grid])

# #%% reading e57 files

# import pye57
# e57 = pye57.E57(r"D:\Uri\Hamlet\Customers\MetaBIM\Data\2023-02-10\Parts\e57\ball_valve_act.e57")
# # the ScanHeader object wraps most of the scan information:
# header = e57.get_header(0)
# print(header.point_count)
# print(header.rotation_matrix)
# print(header.translation)
