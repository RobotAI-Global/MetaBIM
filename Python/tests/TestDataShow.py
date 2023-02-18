

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
#     pip install matplotlib
#     pip install opencv-contrib-python

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
las.header.scales # dimensions

list(las.point_format.dimension_names)
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))



#print(list(las.classification))

#%%
# Some notes on the code below:
# 1. las.header.maxs returns an array: [max x, max y, max z]
# 2. `|` is a numpy method which performs an element-wise "or"
#    comparison on the arrays given to it. In this case, we're interested
#    in points where a XYZ value is less than the minimum, or greater than
#    the maximum.
# 3. np.where is another numpy method which returns an array containing
#    the indexes of the "True" elements of an input array.

# Get arrays which indicate invalid X, Y, or Z values.
X_invalid = (las.header.mins[0] > las.x) | (las.header.maxs[0] < las.x)
Y_invalid = (las.header.mins[1] > las.y) | (las.header.maxs[1] < las.y)
Z_invalid = (las.header.mins[2] > las.z) | (las.header.maxs[2] < las.z)
bad_indices = np.where(X_invalid | Y_invalid | Z_invalid)

print(bad_indices)

#%% basic distance

# Pull off the first point
first_point = point_data[0,:]
# Calculate the euclidean distance from all points to the first point
distances = np.sqrt(np.sum((point_data - first_point) ** 2, axis=1))
# Create an array of indicators for whether or not a point is less than
# 500000 units away from the first point
mask = distances < 800
# Grab an array of all points which meet this threshold
points_kept = las.points[mask]
print("We kept %i points out of %i total" % (len(points_kept), len(las.points)))

#%% Advanced neighborhood

from scipy.spatial import cKDTree
# Build the KD Tree
tree = cKDTree(point_data)

# This should do the same as the FLANN example above, though it might
# be a little slower.
neighbors_distance, neighbors_indices = tree.query(point_data[100], k=5)
print(neighbors_indices)
print(neighbors_distance)

#%% check colors
point_colors = np.vstack((las.red, las.green, las.blue)).transpose()

#%% Extract objects
# buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
# buildings.points = las.points[las.classification == 6]
# building_data = np.stack([buildings.X, buildings.Y, buildings.Z], axis=0).transpose((1, 0))

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(building_data)
# o3d.visualization.draw_geometries([pcd])

#%% show 
import matplotlib.pyplot as plt
plt.hist(distances)
plt.title("Histogram of the Distances")
plt.show()

#%% random downsample
factor=10
point_data_decimated_random = point_data[::factor]

#%% show
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([pcd])

#%% downsample by 0.5
print("Downsample the point cloud with a voxel of 0.5")
downpcd = pcd.voxel_down_sample(voxel_size=2.5)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% downsample
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=100.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% voxels
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=500.40)
o3d.visualization.draw_geometries([voxel_grid],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% +\-\N
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
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

#%% crop point cloud

# #%% Show all
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_data)
# pcd.colors = o3d.utility.Vector3dVector(point_colors/65535)
# pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([pcd])

#%% Chair core 
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

#%% color object
print("Paint chair")
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

#%% crop object
# Load data
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
chair = vol.crop_point_cloud(pcd)

dists = pcd.compute_point_cloud_distance(chair)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_chair = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_chair],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% coloring differnet colors
print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")

#sample_pcd_data = o3d.data.PCDPointCloud()
#pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
demo_crop_data = o3d.data.DemoCropPointCloud()
pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

#%% find closest points
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

print("Paint the 1501st point red.")
pcd.colors[1500] = [1, 0, 0]

print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

print("Find its neighbors with distance less than 0.2, and paint them green.")
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])


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
