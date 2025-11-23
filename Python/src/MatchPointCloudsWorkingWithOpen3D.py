'''
3D Point Cloud Data Matching

Matches 3D datasets from source to target.
Point clouds are in mm.

Installation Env:

    Avital:  conda activate C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Python\envs\metabim
    
Installation Packages:    
    # 
    pip install open3d
    pip install laspy[lazrs,laszip]
    pip install matplotlib
    conda install spyder-kernels=2.3
       

-----------------------------
 Ver	Date	 Who	Descr
-----------------------------
0202   28.04.23  UD     New data for match
0201   14.04.23  UD     simplifying the Matching3D
0102   28.02.23  UD     adding laz file support
0101   25.02.23  UD     Created
-----------------------------

'''

import os
import numpy as np
import open3d as o3d
import laspy
import time
#import cv2 as cv
#import json
import copy
from scipy.spatial.distance import cdist, pdist, squareform
import unittest
import matplotlib.pyplot as plt

plt.ioff()  # interactive mode off

#%% Help functions
# ======================
def get_test_vertex(vtype = 1):
    # define points for tests
    if vtype == 1: # non symmetrical 2 opposite points
        point_data = np.array(
            [
                [-1, -1, -1],
                [-1, -1,  0.5],            
                [-1,  0.6, -1],
                [ 0.7, -1, -1],                               
                [-0.5,  1,  1],   
                [ 1, -0.6,  1],               
                [ 1,  1, -0.7],
                [ 1,  1,  1],             
                #[0, 0, 0],
            ],
            dtype=np.float64,
        )
    elif vtype == 2: # symmetrical
        point_data = np.array(
            [
                [-1, -1, -1],
                [-1, -1,  1],            
                [-1,  1, -1],
                [-1,  1,  1],   
                [ 1, -1, -1],
                [ 1, -1,  1],               
                [ 1,  1, -1],
                [ 1,  1,  1],             
                #[0, 0, 0],
            ],
            dtype=np.float64,
        )
        
    elif vtype == 10: # random with 1 match
        point_data = np.random.rand(10,3)*100       

    elif vtype == 11: # random with 1 match
        point_data = np.random.rand(64,3)*100

    elif vtype == 12: # random with 2 matches
        point_data = np.random.rand(256,3)*100
        point_data = np.vstack((point_data,point_data[::-1,:]+2))

    elif vtype == 13: # random with 2 matches
        point_data = np.random.rand(1024,3)*1000
        point_data = np.vstack((point_data,point_data[::-1,:]+2))
    
    else:
        ValueError('bad vtype')
        
    return point_data

def get_pcd_from_vertex(point_data, clr = [0.3, 0.3, 0.3] ):
    # transform points to pcd

    pcd             = o3d.geometry.PointCloud()
    pcd.points      = o3d.utility.Vector3dVector(point_data)
    pcd.paint_uniform_color(clr)
    #o3d.visualization.draw([pcd])
    return pcd

def apply_noise(points, mu = 0, sigma = 1):
    #noisy_pcd = copy.deepcopy(pcd)
    #points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    #noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return points

def find_closest_points(point_data, point_coord = [0,0,0],  max_dist = 100):
    # find closest points
   # search by radius
    print("Find neighbors of %s point with distance less than %d"  %(str(point_coord),max_dist ))

    pcd      = get_pcd_from_vertex(point_data)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    #[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], point_num)
    #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    
    #print("Find its neighbors with distance less than X, and paint them green.")
    #print("Find its Y nearest neighbors, and paint them blue.")    
    #[k, idx_dist_max, _] = pcd_tree.search_hybrid_vector_3d(np.array(point_coord), radius = max_dist, max_nn = point_num)
    [k, idx_dist_max, _] = pcd_tree.search_radius_vector_3d(point_coord, max_dist)
    #np.asarray(pcd.colors)[idx_dist_max[1:], :] = [0, 1, 0]
    print('Point number %d at max distance %d ' %(len(idx_dist_max),max_dist))
    
    # [k, idx_dist_min, _] = pcd_tree.search_radius_vector_3d(point_coord, min_dist)
    # np.asarray(pcd.colors)[idx_dist_min[1:], :] = [0, 0, 1]    
    # print('Point number %d at min distance %d ' %(len(idx_dist_min),min_dist))

    #[k, idx_num, _] = pcd_tree.search_knn_vector_3d(point_coord, point_num)
    #np.asarray(pcd.colors)[idx_num[1:], :] = [0, 0, 1]
    
    # difference betwee all the indices
    #idx          = np.array(set(idx_dist_max) ^ set(idx_dist_min))
    #idx          = [x for x in idx_dist_max if x not in idx_dist_min[1:]]
    
    # select points
    #idx          = idx[: np.minimum(len(idx),point_num)]
    #pcd_knn      = pcd.select_by_index(idx)
    point_data_indx = point_data[idx_dist_max, :]
    return point_data_indx

def find_closest_points_in_cube(point_data, point_coord = [0,0,0],  max_dist = 100):
    # find closest points by search in cube
    print("Find neighbors of %s point with cube distance less than %d"  %(str(point_coord),max_dist ))
    
    p_bool   = np.ones((point_data.shape[0],1), dtype=bool)
    for k in range(3):
        x_bool   = np.logical_and((point_coord[k]-max_dist < point_data[:,k]) , (point_data[:,k] < point_coord[k]+max_dist))
        p_bool   = np.logical_and(p_bool , x_bool.reshape(p_bool.shape))

    point_data_indx = point_data[p_bool.ravel(),:]
    return point_data_indx

# ======================
def dist2key(d, factor = 10):
    # creates key from distance
    #k = np.round(d*factor)/factor
    k = np.round(d/factor)*factor
    return k
    
def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary, 
        it will associate multiple values with that 
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [dict_obj[key], value]

def get_match_pairs(dist_hash, dist_points):
    # extract matching pairs from hash dictionary
    #dist_points = np.linalg.norm(points[i] - points[j])
    dkey  = dist_points #dist2key(dist_points) #np.round(dist_points*10)/10
    pairs = dist_hash.get(dkey)
    if pairs is not None:
        if len(pairs) > 1000:
            print('Too many matches - consider to reduce bin size. Limiting.')
            pairs = pairs[:1000]
        
    return pairs

def dict_intersect(pairs_ij, pairs_jk):
    # intersect common indices of the sets
    # make all the combinations when the first and last column matches
    
    pairs_ik = {}
    if len(pairs_ij) > 1000 and len(pairs_jk) > 1000:
        print('too many pairs to match')
    
    for sij in pairs_ij:
        for sjk in pairs_jk:
            if sij[1] == sjk[0]:
                prev_node_list = pairs_ij[sij]
                prev_node_list.append(sij[1])
                #prev_node_list = sij[1]
                add_value(pairs_ik,(sij[0] ,sjk[1]),prev_node_list)

    return pairs_ik

def dict_intersect_cycle(cycle_ij, pairs_jk):
    # intersect common indices of the sets
    # make all the combinations when the first and last column matches
    
    cycle_ijk = {}
    if len(pairs_jk) > 1000:
        print('too many pairs to match. bin width is too wide')
    
    for sij in cycle_ij:
        for sjk in pairs_jk:
            if sij[-1] == sjk[0]:
                cycle_key = sij[:-1] + sjk
                cycle_ijk[cycle_key] = 1

    return cycle_ijk


def numpy_vectorized(X,Y):
    return np.sum((X[:,None,:] - Y[None,:,:])**2, axis=2)**0.5

def scipy_cdist(X,Y):
    # a = np.array([[0, 0, 0],
    #           [0, 0, 1],
    #           [0, 1, 0],
    #           [0, 1, 1],
    #           [1, 0, 0],
    #           [1, 0, 1],
    #           [1, 1, 0],
    #           [1, 1, 1]])
    # b = cdist(a,a, metric='euclidean')
    # c = squareform(pdist(a,metric='euclidean'))
    return cdist(X,Y,metric='euclidean')

def assignment_step_v5(data, centroids):
    # nice speed up using einstein summation : https://nicholasvadivelu.com/2021/05/10/fast-k-means/
    diff = data[:, None] - centroids[None]  # (n, k, d)
    distances = np.einsum('nkd,nkd->nk', diff, diff)  # (n, k)
    labels = np.argmin(distances, axis=1)  # (n,)
    return labels

#%% Deals with multuple templates
class MatchPointClouds:
    
    def __init__(self, config = None):
               
        self.state              = False  # object is initialized
        self.cfg                = config
        self.debugOn            = False
        self.figNum             = 1      # control names of the figures
        self.t_start            = time.time()
        
        self.SENSOR_NOISE       = 1      # sensor noise in mm
        self.MAX_OBJECT_SIZE    = 3000   # max object size to be macthed 
        
        self.SRC_DOWNSAMPLE     = 1     # downsample the sourcs model
        self.DST_DOWNSAMPLE     = 1     # downsample the target model
        
        # param
        # DIST_BIN_WIDTH helps to assign multiple distances to the same bin. 
        # higher DIST_BIN_WIDTH more distances in the same bin
        self.DIST_BIN_WIDTH   = 1     # how many right zeros in the distance number
        
        # how point selection is done
        self.POINT_SELECT_TYPE  = 11   # 3-random, 11-max and min
        
        self.src_points         = None   # model points Nx3 to be matched
        self.dst_points         = None   # target points Mx3 to be matched to  
        
        self.src_cycle         = None   # cycle selected
        self.dst_cycle         = None   # cycles detected                
        
        self.srcObj3d           = None   # model to be matched
        self.dstObj3d           = None   # target to be matched to
        
        self.srcDist            = None   # distance hash for source
        self.dstDist            = None   # distance hash for target
        
        self.srcPairs           = None   # pairs hash for source (inverse index of srcDist)
        self.dstPairs           = None   # pairs hash for target (inverse index of dstDist)
        
        # self.srcAdjacency       = None   # adjacency mtrx for source
        # self.dstAdjacency       = None   # adjacency mtrx for target 
        self.dbg_dist_value     = None    # jusy for debug
   
               
        self.Print('3D-Manager is created')
        
    def SelectTestCase(self, testType = 1):        
        # loads 3D data by test specified
        ret = False
        if testType == 1:    
            point_data      = get_test_vertex(1)*10

            self.src_points = point_data + 1  # small shift to see the difference
            self.dst_points = point_data
     
        elif testType == 2:
            point_data      = get_test_vertex(10)
            self.src_points = point_data + 1  # small shift to see the difference
            self.dst_points = point_data
            
        elif testType == 3:
            point_data      = get_test_vertex(11)
            self.src_points = point_data + 1  # small shift to see the difference
            self.dst_points = point_data
            
        elif testType == 4:
            self.Print('two point clouds with different extensions')
            point_data  = get_test_vertex(1)
            self.src_points = point_data  # small shift to see the difference
            point_data  = get_test_vertex(2)
            self.dst_points = point_data 
            
        elif testType == 11:
            self.Print("one point cloud is a part of the other.")
            point_data  = get_test_vertex(11)
            self.dst_points = point_data
            point_data  = point_data[:27,:]
            self.src_points = point_data + 1  # small shift to see the difference
            
        elif testType == 12:
            self.DIST_BIN_WIDTH   = 0.1
            self.Print("one point cloud is a part of the other .")
            point_data  = get_test_vertex(12)
            self.dst_points = point_data
            point_data  = point_data[:7,:]
            self.src_points = point_data + 1  # small shift to see the difference        

        elif testType == 13:
            self.Print("one point cloud is a part of the other.")
            self.DIST_BIN_WIDTH   = 0.1  # working
            point_data  = get_test_vertex(13)
            self.dst_points = point_data
            point_data  = point_data[:37,:]
            self.src_points = point_data + 1  # small shift to see the difference  
            
        elif testType == 14:
            self.Print("one point cloud is a part of the other - different bin width.")
            self.DIST_BIN_WIDTH   = 0.05  # 1, 10 -is too many matches, 0.01 is good but slow
            point_data  = get_test_vertex(13)
            self.dst_points = point_data
            point_data  = point_data[:37,:]
            self.src_points = point_data + 5  # small shift to see the difference  
            
            self.debugOn = False        
            
        elif testType == 15:
            self.Print("one point cloud is a part of the other - noise.")
            self.DIST_BIN_WIDTH = 0.1  # 1, 10 -is too many matches, 0.01 is good but slow
            point_data          = get_test_vertex(12)
            self.dst_points     = point_data
            point_data          = point_data[:51,:]
            self.src_points     = apply_noise(point_data,5,9)  # small shift to see the difference  
            
            self.debugOn = False                     
                                                          
        elif testType == 31:
            self.Print("Load customer models.")
            dataPath = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\valve.laz')
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/1000
            las             = laspy.read(dataPath + '\\valve.laz') # valve pressure # farm2M
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.src_points = point_data/1000 + 1  # small shift to see the difference 
                        
            self.SRC_DOWNSAMPLE     = 50     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 50     # downsample the target model  
            self.DIST_BIN_WIDTH     = 1 
            
        elif testType == 41:
            
            self.SENSOR_NOISE       = 10      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 2000   # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 10 
            self.SRC_DOWNSAMPLE     = 400     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 400     # downsample the target model 
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            dst_scale       = np.max(las.header.scales)
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/1
            self.Print('Model point numbers : %d' %(point_data.shape[0]))  
            self.Print('Dimensions  : %s' %str(point_data.max(0) - point_data.min(0)))
            
            self.Print("Load Search model - old data...")
            #dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-04-24'
            las             = laspy.read(dataPath + '\\pump.laz') # valve pressure # farm2M
            self.Print('Model scale : %s' %str(las.header.scales)) # dimensions
            src_scale       = np.max(las.header.scales)
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            point_data      = point_data * src_scale /dst_scale + 0
            self.src_points = point_data + 10  # small shift to see the difference 
            self.Print('Model point numbers : %d' %(point_data.shape[0]))
            self.Print('Dimensions  : %s' %str(point_data.max(0) - point_data.min(0)))
                
            
        elif testType == 42:
            
            self.SENSOR_NOISE       = 1      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 2000   # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 10 
            self.SRC_DOWNSAMPLE     = 1     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 300     # downsample the target model  
                        
            self.Print("Load Target model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Model scale : %s' %str(las.header.scales)) # dimensions
            dst_scale       = np.max(las.header.scales)
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/1+3
            self.Print('Model point numbers : %d' %(point_data.shape[0]))  
            self.Print('Dimensions  : %s' %str(point_data.max(0) - point_data.min(0)))
            
            self.Print("Load Search model - new data...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-04-24'
            las             = laspy.read(dataPath + '\\pump.laz') # valve pressure # farm2M
            self.Print('Model scale : %s' %str(las.header.scales)) # dimensions
            src_scale       = np.max(las.header.scales)
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            point_data      = point_data * src_scale /dst_scale + 0
            self.src_points = point_data + 0  # small shift to see the difference 
            self.Print('Model point numbers : %d' %(point_data.shape[0]))
            self.Print('Dimensions  : %s' %str(point_data.max(0) - point_data.min(0)))
            
            self.Print("Load models is done.")      
                   
          
        elif testType == 51: # selection from the small model
                        
            self.SENSOR_NOISE       = 1      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 500       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 1
            self.SRC_DOWNSAMPLE     = 50     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 50     # downsample the target model            
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\valve.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/10000
            
            self.Print("Select search model...")
            point_data_subset = find_closest_points(self.dst_points, point_coord = self.dst_points[1000,:],  max_dist = self.MAX_OBJECT_SIZE/2)
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 1  # small shift to see the difference
             
            self.Print("Load models is done.")      
            # need to adjust size   
            
        elif testType == 52: # selection from the small model
                        
            self.SENSOR_NOISE       = 1      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 500       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 10
            self.SRC_DOWNSAMPLE     = 40     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 40     # downsample the target model            
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\valve.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/10000
            
            self.Print("Select search model...")
            point_data_subset = self.dst_points[:1000,:]
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 1  # small shift to see the difference
             
            self.Print("Load models is done.")  
                        
        elif testType == 61: # selection from the big model
                        
            self.SENSOR_NOISE       = 10      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 5000       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 10 
            self.SRC_DOWNSAMPLE     = 200     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 200     # downsample the target model            
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/1
            
            self.Print("Select search model...")
            point_data_subset = find_closest_points_in_cube(point_data, point_coord = point_data[700000,:],  max_dist = self.MAX_OBJECT_SIZE/2)
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 5  # small shift to see the difference
             
            self.Print("Load models is done.")      
            # need to adjust size
            
        elif testType == 62: # selection from the big model
                        
            self.SENSOR_NOISE       = 100      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 30000       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 100 
            self.SRC_DOWNSAMPLE     = 500     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 500     # downsample the target model               
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data/1
            
            self.Print("Select search model...")
            #point_data_subset = self.dst_points[1750000:1800000,:]  # good
            #point_data_subset = self.dst_points[1840000:1850000,:] # good
            point_data_subset = self.dst_points[1800000:1870000,:] # good
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 5  # small shift to see the difference
             
            self.Print("Load models is done.") 
            
        elif testType == 71: # selection from the big model and then smaller model
                        
            self.SENSOR_NOISE       = 10      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 1000       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 1
            self.SRC_DOWNSAMPLE     = 1     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 1     # downsample the target model            
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            point_data      = find_closest_points_in_cube(point_data, point_coord = point_data[1600000,:],  max_dist = self.MAX_OBJECT_SIZE/2)
            self.dst_points = point_data/1
            
            self.Print("Select search model...")
            point_data_subset = find_closest_points_in_cube(point_data, point_coord = point_data[100,:],  max_dist = self.MAX_OBJECT_SIZE/4)
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 5  # small shift to see the difference
             
            self.Print("Load models is done.")  
            
        elif testType == 72: # selection from the big model and then smaller model
                        
            self.SENSOR_NOISE       = 10      # sensor noise in mm
            self.MAX_OBJECT_SIZE    = 1000       # max object size to be macthed 
            self.DIST_BIN_WIDTH     = 1
            self.SRC_DOWNSAMPLE     = 1     # downsample the sourcs model
            self.DST_DOWNSAMPLE     = 1     # downsample the target model       
            self.POINT_SELECT_TYPE  = 3     
            
            self.Print("Load reference model...")
            dataPath        = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\farm2M.laz')
            self.Print('Target model scale : %s' %str(las.header.scales)) # dimensions
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            point_data      = find_closest_points_in_cube(point_data, point_coord = point_data[1600000,:],  max_dist = self.MAX_OBJECT_SIZE/2)
            self.dst_points = point_data/1
            
            self.Print("Select search model...")
            point_data_subset = find_closest_points_in_cube(point_data, point_coord = point_data[100,:],  max_dist = self.MAX_OBJECT_SIZE/4)
            if point_data_subset.shape[0] < 100:
                raise ValueError("Can not selectr enougph points")
                
            self.src_points = point_data_subset/1 + 5  # small shift to see the difference
             
                    
                             
        else:
            self.Print('Bad choice %d is selected' %testType, 'E')
            return ret    
        
        # output
        self.Print('Data set %d is selected' %testType, 'I')
        return True    
    
    def Downsample(self, points, factor = 300):
        # reduce num of points
        #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=100.5)
        point_num   = points.shape[0]
        
        factor      = np.maximum(factor, int(point_num/3000))
        
        sample_rate = factor #int(len(points)/factor)
        if sample_rate > 1:
            down_points = points[::sample_rate,:]
            self.Print('Downsampling data by factor %d' %sample_rate)
        else:
            down_points = points
            self.Print("No Downsample of the point cloud")
            
        #o3d.visualization.draw([down_pcd])
        return down_points
    
    def MakeCompact(self, points):
        # make storage efficient      
        points = points.astype(np.int32)
        return points

    def PrepareDataset(self, points, min_dist_value = 0, max_dist_value = 1000):
        # compute distances only on the part
        dist_ij     = scipy_cdist(points,points)
        # dist_ij[dist_ij < min_dist_value] = 1e9
        # dkey_ij     = dist2key(dist_ij,10)
        # index_ij    = dkey_ij.argsort(axis = 1)
        # knn_num     = np.minimum(knn_num,len(points))
        self.Print('Statistics:  Point number: %5d' %(dist_ij.shape[0]))
        self.Print('Statistics: Min-Max range: %4.1f - %4.1f' %(min_dist_value, max_dist_value ))
        self.Print('Statistics:  Max distance: %4.1f' %(dist_ij.max()))

        #if self.DIST_BIN_WIDTH < 1:
        #    raise ValueError('self.DIST_BIN_WIDTH must be >= 1')
        
        #
        dist_ij               = dist2key(dist_ij, self.DIST_BIN_WIDTH)
        if self.debugOn:
            self.dbg_dist_value   = dist_ij
        
        index_i, index_j      = np.nonzero(np.logical_and(min_dist_value < dist_ij , dist_ij < max_dist_value))
    
        dist_dict = {}
        pairs_dict = {}
        for i,j in zip(index_i, index_j):

            add_value(dist_dict, dist_ij[i,j], (i,j))
            add_value(pairs_dict,(i,j), dist_ij[i,j])

        return dist_dict, pairs_dict

    def PreprocessPointData(self, points, factor = 1000):
        # computes hash functions for source and target
        points          = self.Downsample(points, factor)
        
        # make it compact
        points          = self.MakeCompact(points)
        
        # minimal distance between points
        min_dist        = self.SENSOR_NOISE * 10   
        max_dist        = self.MAX_OBJECT_SIZE
        
        # compute different structures from distance
        t_start      = time.time()
        dist_dict, pairs_dict   = self.PrepareDataset(points,  min_dist, max_dist)
        t_stop      = time.time()
        self.Print('Dist Hash time : %4.3f [sec]' %(t_stop - t_start))
        
        return points,  dist_dict, pairs_dict
    
    def SelectMatchPoints(self, points, selectType = 1):
        # select the points according to different criteria   
        point_num = points.shape[0]
        indx = [1,2,3]
        if selectType == 1:
            # definien
            indx = [1,2,3]
           
        elif selectType == 2:
             # random
            indx = np.random.randint(0, point_num, size=3)
            
        elif selectType == 3:
             # random with multiple trials
            if self.srcPairs is None:
                self.Peint('init dataset')
                return 
            
            # create cycles at randmom and check validity
            trial_num = 90
            while trial_num > 0:
                trial_num = trial_num -1
                indx = np.random.randint(0, point_num, size=3) 
                allin = (indx[0],indx[1]) in self.srcPairs
                allin = (indx[1],indx[2]) in self.srcPairs and allin
                allin = (indx[2],indx[0]) in self.srcPairs and allin
                if allin:
                    break   
             
            if not allin:
                self.Print('Failed to find a cycle','W')        
                  
        elif selectType == 11:
             # min max elements of the array - most distant points
            indx[0] = np.argmax(points[:,0], 0)
            indx[1] = np.argmin(points[:,0], 0)
            indx[2] = np.argmax(points[:,1], 0)
         
        elif selectType == 12:
             # min max elements of the array - most distant points
            dist    = np.dot(points,np.array([1,1,1]))
            indx[0] = np.argmax(dist, 0)
            indx[1] = np.argmin(dist, 0)
            dist    = np.dot(points,np.array([1,1,-1]))
            indx[2] = np.argmax(dist, 0)           
        elif selectType == 21:
            # find the closest points for selection
            min_dist   = (d.srcE.max()/100)**2
            max_dist   = (d.srcE.max()/1)**2
            pcd_knn, indx  = find_closest_points(d.srcObj3d, point_coord = d.srcC, min_dist = min_dist, max_dist = max_dist, point_num = 3)
            
        else:
            indx = [1,2,3]
            self.Print('Bad selectType','E')
            
        return indx
           
    def MatchCycle3(self, indx = []):
        # match in several steps
        #if self.dstAdjacency is None:
        #    self.Preprocess()

        if len(indx) < 3:
            # extract distances for the designated points
            snodes      = [1,2,3]
        else:
            snodes      = list(indx)
            
        self.Print('Index : %s' %str(snodes))
            
        sshift      = np.roll(snodes,-1) #snodes[1:] + snodes[:1]  # shift nodes by 1
        spairs      = [(snodes[k],sshift[k]) for k in range(len(snodes))]

        # move over from point to point and store all possible matches
        t_start     = time.time()
        cycle_list  = []
        count_cycle = 0
        for sjk in spairs:
            self.Print(sjk)
            dist_jk        = self.srcPairs[sjk]  # extract distance
            dpairs_jk      = get_match_pairs(self.dstDist, dist_jk)
            if dpairs_jk is None:
                dpairs_ij = None
                break
            # init ij - first point
            if count_cycle == 0:
                # use dictionary to trace cycles
                dpairs_ij = {djk:[djk[0]] for djk in dpairs_jk}  # starting point
            else:
                dpairs_ij = dict_intersect(dpairs_ij, dpairs_jk)
            # store pairs for traceback
            cycle_list.append(dpairs_ij)
            count_cycle += 1 
            #print("set dij:\n",dpairs_ij)
        
        # check
        if dpairs_ij is None:
            self.Print('Failed to find cycles')
            return False
            
        # extract closed cycles
        dpairs_ij = cycle_list[-1]
        dpairs_ii = {dii:dpairs_ij[dii] for dii in dpairs_ij if dii[0]==dii[1]}
        
        spairs_ii = {(snodes[0],snodes[0]) : snodes}
        print("scycle dii:",spairs_ii)
        print("dcycle dii:",dpairs_ii)
        t_stop      = time.time()
        self.Print('Match time : %4.3f [sec]' %(t_stop - t_start))
            
        self.src_cycle         = spairs_ii   # cycle selected
        self.dst_cycle         = dpairs_ii   # cycles detected     
            
        isDetected             = len(dpairs_ii) > 0
        
        return isDetected
        
    def MatchCycleDict3(self, indx = []):
        # match in several steps
        #if self.dstAdjacency is None:
        #    self.Preprocess()

        if len(indx) < 3:
            # extract distances for the designated points
            snodes      = [1,2,3]
        else:
            snodes      = list(indx)
            
        self.Print('Index : %s' %str(snodes))
        
        
        sshift      = np.roll(snodes,-1) #snodes[1:] + snodes[:1]  # shift nodes by 1
        spairs      = [(snodes[k],sshift[k]) for k in range(len(snodes))]
        
        # check the index are different points
        isGood      = np.all([skj[0] != skj[1] for skj in spairs])
        if not isGood:
            self.Print('Provided indexes of the match mode are not good %s' %str(snodes))
            return False        

        # move over from point to point and store all possible matches
        t_start     = time.time()
        cycle_list  = []
        count_cycle = 0
        for sjk in spairs:
            self.Print(sjk)
            dist_jk        = self.srcPairs[sjk]  # extract distance
            dpairs_jk      = get_match_pairs(self.dstDist, dist_jk)
            if dpairs_jk is None:
                cycle_ij = None
                break
            # init ij - first point
            if count_cycle == 0:
                # use dictionary to trace cycles
                cycle_ij = {djk:1 for djk in dpairs_jk}  # starting point
            else:
                cycle_ij = dict_intersect_cycle(cycle_ij, dpairs_jk)
            # store pairs for traceback
            #cycle_list.append(dpairs_ij)
            count_cycle += 1 
            #print("set cycle_ij:\n",cycle_ij)
            self.Print("set cycle_ij length %d:" %len(cycle_ij))
        
        # check
        if cycle_ij is None:
            self.Print('Failed to find cycles')
            return False
            
        # extract closed cycles
        #dpairs_ij = cycle_ij #cycle_list[-1]
        dpairs_ii = {dii[:-1]:1 for dii in cycle_ij if dii[0]==dii[-1]}
        
        #spairs_ii = {(snodes[0],snodes[0]) : snodes}
        spairs_ii = {(snodes[0],snodes[1],snodes[2]) : 1}
        print("scycle dii:",spairs_ii)
        print("dcycle dii:",dpairs_ii)
        t_stop      = time.time()
        self.Print('Match time : %4.3f [sec]' %(t_stop - t_start))
            
        self.src_cycle         = spairs_ii   # cycle selected
        self.dst_cycle         = dpairs_ii   # cycles detected     
            
        isDetected             = len(dpairs_ii) > 0
        if not isDetected :
            self.Print('No detection. Consider to increase DIST_BIN_WIDTH or reduce DOWNSAMPLING','W')
        return isDetected
            
    def MatchSourceToTarget(self, src_points = None, dst_points = None):
        # main processing and matching
        if src_points is None or dst_points is None:
            src_points, dst_points = self.src_points, self.dst_points
            
        # factor 1000 make min distance small and keeps all the points
        self.Print('Target model...')
        dst_points, dst_dist_dict, dst_pairs_dict  = self.PreprocessPointData(dst_points, self.DST_DOWNSAMPLE)     
        self.ShowDistanceHistogram(self.dbg_dist_value,'Dst Distances')
        self.ShowDictionaryHistogram(dst_dist_dict,'Dst Dictionary')
         
        # preprocess all the dta
        self.Print('Match model...')
        src_points, src_dist_dict, src_pairs_dict  = self.PreprocessPointData(src_points, self.SRC_DOWNSAMPLE)
        self.ShowDistanceHistogram(self.dbg_dist_value,'Src Distances')
        self.ShowDictionaryHistogram(src_dist_dict,'Src Dictionary')
    
        # need to do selection
        self.srcPairs   = src_pairs_dict
        self.dstDist    = dst_dist_dict     
         
        # select indexes
        src_indx        = self.SelectMatchPoints(src_points, selectType = self.POINT_SELECT_TYPE)

        # match according to indexes
        #isOk            = self.MatchCycle3(src_indx)         
        isOk            = self.MatchCycleDict3(src_indx) 

        
        return True
        
    def ComputeTransformation(self, source, target, picked_id_source, picked_id_target):
        # point to point transformation using indexes
            
        # pick points from two point clouds and builds correspondences
        #picked_id_source = pick_points(source)
        #picked_id_target = pick_points(target)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr        = np.zeros((len(picked_id_source), 2))
        corr[:, 0]  = picked_id_source
        corr[:, 1]  = picked_id_target
        
        #print(corr)
        print(np.array(source.points)[picked_id_source,:] - np.array(target.points)[picked_id_target,:])

        # estimate rough transformation using correspondences
        self.Print("Compute a rough transform using the correspondences ")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))  
        #print(trans_init)
        
        # point-to-point ICP for refinement
        self.Print("Perform point-to-point ICP refinement")
        threshold = 0.03  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())        
        
        #source_trasformed = self.draw_registration_result(source, target, reg_p2p.transformation) 
        
        source_temp = copy.deepcopy(source)
        #target_temp = copy.deepcopy(target)
        #source_temp.paint_uniform_color([1, 0.706, 0])
        #target_temp.paint_uniform_color([0, 0.651, 0.929])
        
        #source_temp.transform(reg_p2p.transformation)
        print(reg_p2p.transformation)
        
        #o3d.visualization.draw_geometries([source_temp, target_temp])
        source_trasformed = source_temp
        
        # debug
        src_points = np.asarray(source.points)[picked_id_source,:]
        dst_points = np.asarray(target.points)[picked_id_target,:]
        #print(src_points, dst_points )
        #print(reg_p2p.transformation)
        return source_trasformed 
    
    def ShowMatchedData3D(self, wname = 'Source'):
        # show 3D data with multiple matcing sources
        ret = False
        if self.src_points is None or self.dst_points is None:
            self.Print('Point cloud data is not specified','W')
            return
        
        src_indx = [k for k, v in self.src_cycle.items()]
        dst_indx = [k for k, v in self.dst_cycle.items()]
        dst_match_num = len(dst_indx)
        if dst_match_num < 1:
            self.Print('No matcing points are found','W')
            return 
        
        if dst_match_num > 3:
            dst_match_num = 3  
            self.Print('Too many matches are found. Showing first 3','I')    
            
        clrs    = [[0.8, 0.1, 0.1],[0.1, 0.1, 0.8], [0.8, 0.1, 0.8]]    
        
        # convert
        dst_pcd = get_pcd_from_vertex(self.dst_points, clr = [0.1, 0.8, 0.1] )
        src_pcd = get_pcd_from_vertex(self.src_points, clr = [0.8, 0.1, 0.1] )
        
        # start matching
        pcd_list = [dst_pcd]
        
        for m in range(dst_match_num):
            s_indx = src_indx[0]
            d_indx = dst_indx[m]
            srd_pcd_trasformed = self.ComputeTransformation(src_pcd, dst_pcd, s_indx, d_indx)
            srd_pcd_trasformed.paint_uniform_color(clrs[m])
            pcd_list.append(srd_pcd_trasformed)
            

        #source_temp.paint_uniform_color([0.9, 0.1, 0])
        #target_temp.paint_uniform_color([0, 0.9, 0.1])
        #source_temp.transform(transformation)
        o3d.visualization.draw(pcd_list)
        #o3d.visualization.draw_geometries([source_temp, target_temp], window_name = wname)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                               zoom=0.4559,
        #                               front=[0.6452, -0.3036, -0.7011],
        #                               lookat=[1.9892, 2.0208, 1.8945],
        #                               up=[-0.2779, -0.9482, 0.1556])      
        
        # aabb = src_pcd.get_axis_aligned_bounding_box()
        # aabb.color = (1, 0, 0)
        # obb = chair.get_oriented_bounding_box()
        # obb.color = (0, 1, 0)
        # o3d.visualization.draw_geometries([src_pcd, aabb, obb],
        #                                 zoom=0.7,
        #                                 front=[0.5439, -0.2333, -0.8060],
        #                                 lookat=[2.4615, 2.1331, 1.338],
        #                                 up=[-0.1781, -0.9708, 0.1608])
        

        return True    
        
    def ColorSpecificPoints(self, pcd, pairs_ij, clr = [0, 1, 0]):   
        self.Print('Colors specific points and the rest are gray') 
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if pairs_ij is None:
            return pcd
        
        idx = [s[0] for s in pairs_ij]
        np.asarray(pcd.colors)[idx, :] = clr
        
        return pcd
    
    def ColorCycles(self, pcd, cycle_ii, clr = [0, 1, 0]):   
        # cycle_ii is a dictionary with cycle per point
        #self.Print('Colors point cycles and the rest are gray') 
        #
        if cycle_ii is None:
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
            return pcd
        
        for p in cycle_ii.keys():
            idx = p #cycle_ii[p]
            np.asarray(pcd.colors)[idx, :] = clr
        
        return pcd
        
    def ShowData3D(self, wname = 'Source'):
        # show 3D data
        ret = False
        # if not isinstance(src, type(o3d)):
        #     self.Print('Need open3d object - src')
        #     return ret
        # if not isinstance(dst, type(o3d)):
        #     self.Print('Need open3d object - dst')
        #     return ret
                
        if self.src_points is None or self.dst_points is None:
            self.Print('Point cloud data is not specified','W')
            return
        
        # convert
        dst_pcd = get_pcd_from_vertex(self.dst_points, clr = [0.1, 0.8, 0.1] )
        src_pcd = get_pcd_from_vertex(self.src_points, clr = [0.8, 0.1, 0.1] )
        
       
        # color cycles
        dst_pcd = self.ColorCycles(dst_pcd, self.dst_cycle, [0,1,1])
        src_pcd = self.ColorCycles(src_pcd, self.src_cycle, [1,0,1])
            


        #source_temp.paint_uniform_color([0.9, 0.1, 0])
        #target_temp.paint_uniform_color([0, 0.9, 0.1])
        #source_temp.transform(transformation)
        o3d.visualization.draw([src_pcd, dst_pcd])
        #o3d.visualization.draw_geometries([source_temp, target_temp], window_name = wname)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                               zoom=0.4559,
        #                               front=[0.6452, -0.3036, -0.7011],
        #                               lookat=[1.9892, 2.0208, 1.8945],
        #                               up=[-0.2779, -0.9482, 0.1556])      
        
        # aabb = src_pcd.get_axis_aligned_bounding_box()
        # aabb.color = (1, 0, 0)
        # obb = chair.get_oriented_bounding_box()
        # obb.color = (0, 1, 0)
        # o3d.visualization.draw_geometries([src_pcd, aabb, obb],
        #                                 zoom=0.7,
        #                                 front=[0.5439, -0.2333, -0.8060],
        #                                 lookat=[2.4615, 2.1331, 1.338],
        #                                 up=[-0.1781, -0.9708, 0.1608])
        

        return True

    def ShowDistanceHistogram(self, distance_values = None, title_text = ''):
        # displays distance histogram
        if not self.debugOn:
            return
        
        if distance_values is None:
            if self.dbg_dist_value is None:
                return
            else:
                distance_values = self.dbg_dist_value
                self.Print('Debug mode is on')
        
        
        #low_bound = np.percentile(distance_values, 2)
        #top_bound = np.percentile(distance_values, 98)
        low_bound = np.min(distance_values)
        top_bound = np.max(distance_values)
        bin_number = int((top_bound - low_bound)/self.DIST_BIN_WIDTH)
        his_val, bin_edges = np.histogram(distance_values, bins=bin_number, range=(low_bound,top_bound), density=False)
        hist_loc = (bin_edges[:-1] + bin_edges[1:]) / 2
        width    = 0.8 * (bin_edges[1] - bin_edges[0])
        
        plt.figure()
        plt.bar(hist_loc,his_val, align='center', width=width)
        plt.title(title_text + ' bins : ' + str(bin_number))
        plt.show(block=False)
        
    def ShowDictionaryHistogram(self, dist_dict, title_text = ''):
        # displays distance histogram stored in hash
        if not self.debugOn:
            return
        
        if not isinstance(dist_dict, dict):
            return
        self.Print('Debug mode is on')
        
        distance_values = dist_dict.keys()
        distance_counts = [len(val) for val in dist_dict.values()]
        
        his_val  = distance_counts
        hist_loc = distance_values
        bin_number = len(distance_counts)
        width    = 0.8 * self.DIST_BIN_WIDTH
        
        plt.figure()
        plt.bar(hist_loc, his_val, align='center', width=width)
        plt.title(title_text + ' bins : ' + str(bin_number))
        plt.show(block=False)        

    def Finish(self):
        
        if self.debugOn:
            plt.show()
            
        #cv.destroyAllWindows() 
        self.Print('3D-Manager is closed')
        
    def Print(self, txt='',level='I'):
        t_stop = time.time()
        print('%s: 3DM: %4.3f: %s' %(level, t_stop-self.t_start, txt))
        self.t_start = time.time()

        
       
#%% --------------------------           
class TestMatchPointClouds(unittest.TestCase):                
    def test_Create(self):
        d       = MatchPointClouds()
        self.assertEqual(False, d.state)
        
    def test_LoadData(self):
        # check read training files  
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(1)
        self.assertTrue(isOk)
        
    def test_ShowData(self):
        # check data show  
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(3)
        d.ShowData3D()
        #isOk        = d.SelectTestCase(2)
        #d.ShowData3D(d.srcObj3d,d.dstObj3d, wname = 'Case 2')
        self.assertTrue(isOk)  
        
    def test_Distance(self):
        # testing didtance performance
        #M = np.arange(6000*3, dtype=np.float64).reshape(6000,3)
        M           = np.random.rand(10000,3)
        print('Calculating the distance...')
        t_start     = time.time()
        sp_result   = scipy_cdist(M, M) #Scipy usage
        t_stop      = time.time()
        print('Distance time : %4.3f [sec]' %(t_stop - t_start))
            
    def test_Downsample(self):
        # test trasnformation  
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(11)
        srcDown     = d.Downsample(d.srcObj3d)
        transform   = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])
        d.ShowData3D(d.srcObj3d,srcDown, transform)
        self.assertTrue(isOk)  
        
    def test_Histogram(self):
        # test distance histograms
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(13)  # 62-ok 
        
        d.Print('Target model...')
        dst_points, dst_dist_dict, dst_pairs_dict  = d.PreprocessPointData(d.dst_points, d.DST_DOWNSAMPLE)     
        d.ShowDistanceHistogram(d.dbg_dist_value,'Dst Distances')
        d.ShowDictionaryHistogram(dst_dist_dict,'Dst Dictionary')
        
        # preprocess all the dta
        d.Print('Match model...')
        src_points, src_dist_dict, src_pairs_dict  = d.PreprocessPointData(d.src_points, d.SRC_DOWNSAMPLE)  
        d.ShowDistanceHistogram(d.dbg_dist_value, 'Src Distances')
        d.ShowDictionaryHistogram(src_dist_dict,'Src Dictionary')
                   
        plt.show()
                   
    def test_MatchSourceTarget(self):
        # match cycle 
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(41)  # 62-ok, 41-ok, 11,12-ok, 14-ok, 15-ok,31-ok, 42-ok, 51,52-ok, 61-ok, 71-ok
    
        isOk        = d.MatchSourceToTarget()

        #d.ShowData3D()
        d.ShowMatchedData3D()
        d.Finish()
        self.assertTrue(isOk) 
        



    
               

#%%
if __name__ == '__main__':
    #print (__doc__)
    
    #unittest.main()
    
    # template manager test
    singletest = unittest.TestSuite()
#    singletest.addTest(TestMatching3D("test_Create"))
#    singletest.addTest(TestMatching3D("test_LoadData"))
    singletest.addTest(TestMatchPointClouds("test_ShowData"))  
#    singletest.addTest(TestMatching3D("test_Distance"))
    #singletest.addTest(TestMatchPointClouds("test_MatchSourceTarget"))
#    singletest.addTest(TestMatchPointClouds("test_Histogram"))
    
    unittest.TextTestRunner().run(singletest)