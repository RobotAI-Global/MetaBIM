'''
3D Data Matching

Matches 3D datasets from source to target

Installation Env:

    Avital:  conda activate C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Python\envs\metabim
    
Installation Packages:    
    # 
    pip install open3d
    pip install xgboost
    pip install laspy[lazrs,laszip]
       

-----------------------------
 Ver	Date	 Who	Descr
-----------------------------
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
        point_data = np.random.rand(10,3)*10        

    elif vtype == 11: # random with 1 match
        point_data = np.random.rand(64,3)*100

    elif vtype == 12: # random with 2 matches
        point_data = np.random.rand(256,3)*1000
        point_data = np.vstack((point_data,point_data[::-1,:]+2))
    
    else:
        ValueError('bad vtype')
        
    return point_data

def get_pcd_from_vertex(point_data):
    # transform points to pcd

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])
    #o3d.visualization.draw([pcd])
    return pcd

def apply_noise(pcd, mu = 0, sigma = 1):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

# ======================
def dist2key(d, factor = 10):
    # creates key from distance
    k = np.round(d*factor)/factor
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
      
def distance_hash_python(points):
    # compute distances
    dist_dict = {}
    pairs_dict = {}
    for i in range(len(points)):
        for j in range(len(points)):
        
            dist_ij = np.linalg.norm(points[i] - points[j])
            dkey = dist2key(dist_ij) #  
            add_value(dist_dict, dkey, (i,j))
            add_value(pairs_dict,(i,j), dkey)

    return dist_dict, pairs_dict

def distance_hash_python_fast(points):
    # compute distances
    dist_ij = scipy_cdist(points,points)
    dkey_ij = dist2key(dist_ij,100)
    dist_dict = {}
    pairs_dict = {}
    for i in range(len(points)):
        for j in range(len(points)):
        
            #dist_ij = np.linalg.norm(points[i] - points[j])
            #dkey = dist2key(dist_ij) #  
            add_value(dist_dict, dkey_ij[i,j], (i,j))
            add_value(pairs_dict,(i,j), dkey_ij[i,j])

    return dist_dict, pairs_dict

def distance_hash_python_knn(points, min_dist_value = 0, knn_num = 3):
    # compute distances only on the part
    dist_ij     = scipy_cdist(points,points)
    dist_ij[dist_ij < min_dist_value] = 1e9
    dkey_ij     = dist2key(dist_ij,10)
    index_ij    = dkey_ij.argsort(axis = 1)
    knn_num     = np.minimum(knn_num,len(points))
   
    dist_dict = {}
    pairs_dict = {}
    for i in range(len(points)):
        for js in range(knn_num):
        
            j    = index_ij[i,js]
            add_value(dist_dict, dkey_ij[i,j], (i,j))
            add_value(pairs_dict,(i,j), dkey_ij[i,j])

    return dist_dict, pairs_dict


def adjacency_matrix(edge_list, num_of_nodes):
    # Basic constructor method :  Convert edge list to adjacency list
    
    # # represented with a multi-dimensional array
    # adjacency_mtrx = [[] for _ in range(num_of_nodes)]
    # # Add edges to corresponding nodes of the graph
    # for (origin, dest) in edge_list:
    #     adjacency_mtrx[origin].append(dest)
        
    # represented with a dictionalry
    adjacency_mtrx = {}
    # Add edges to corresponding nodes of the graph
    for (origin, dest) in edge_list:
        add_value(adjacency_mtrx, origin, dest)       
        
    return adjacency_mtrx

def get_match_pairs(dist_hash, dist_points):
    # extract matching pairs from hash dictionary
    #dist_points = np.linalg.norm(points[i] - points[j])
    dkey  = dist2key(dist_points) #np.round(dist_points*10)/10
    pairs = dist_hash.get(dkey)
    return pairs

def dict_intersect(pairs_ij, pairs_jk):
    # intersect common indices of the sets
    # make all the combinations when the first and last column matches
    
    pairs_ik = {}
    
    for sij in pairs_ij:
        for sjk in pairs_jk:
            if sij[1] == sjk[0]:
                prev_node_list = pairs_ij[sij]
                prev_node_list.append(sij[1])
                #prev_node_list = sij[1]
                add_value(pairs_ik,(sij[0] ,sjk[1]),prev_node_list)

    return pairs_ik


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

def find_closest_points(pcd, point_coord = [0,0,0], min_dist = 1, max_dist = 100, point_num = 10):
    # find closest points
   # search by radius
    print("Find %d neighbors of %s point with distance less than %d"  %(point_num,str(point_coord),max_dist ))

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    #[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], point_num)
    #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    
    #print("Find its neighbors with distance less than X, and paint them green.")
    #print("Find its Y nearest neighbors, and paint them blue.")    
    #[k, idx_dist_max, _] = pcd_tree.search_hybrid_vector_3d(np.array(point_coord), radius = max_dist, max_nn = point_num)
    [k, idx_dist_max, _] = pcd_tree.search_radius_vector_3d(point_coord, max_dist)
    np.asarray(pcd.colors)[idx_dist_max[1:], :] = [0, 1, 0]
    print('Point number %d at max distance %d ' %(len(idx_dist_max),max_dist))
    
    [k, idx_dist_min, _] = pcd_tree.search_radius_vector_3d(point_coord, min_dist)
    np.asarray(pcd.colors)[idx_dist_min[1:], :] = [0, 0, 1]    
    print('Point number %d at min distance %d ' %(len(idx_dist_min),min_dist))

    #[k, idx_num, _] = pcd_tree.search_knn_vector_3d(point_coord, point_num)
    #np.asarray(pcd.colors)[idx_num[1:], :] = [0, 0, 1]
    
    # difference betwee all the indices
    #idx          = np.array(set(idx_dist_max) ^ set(idx_dist_min))
    idx          = [x for x in idx_dist_max if x not in idx_dist_min[1:]]
    
    # select points
    idx          = idx[: np.minimum(len(idx),point_num)]
    pcd_knn      = pcd.select_by_index(idx)
    
    return pcd_knn, idx

def kmeans():
    
    from sklearn.cluster import KMeans
    import numpy as np

    # Generate some sample data
    X = np.random.rand(100, 2)

    # Initialize KMeans object with 3 clusters
    kmeans = KMeans(n_clusters=3)

    # Fit KMeans object to the data
    kmeans.fit(X)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the centroids
    centroids = kmeans.cluster_centers_
    
    return centroids

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

def kClosestQuick(points, k) :
    #Solution 2, quick select, Time worst O(n^2) average O(n), Space worst O(n) average O(logn)
    n =  len(points)
    low = 0
    high = n - 1
    while (low <= high) :
        mid = partition(points, low, high)
        if mid == k:
            break
        if (mid < k): 
            low = mid + 1 
        else :
            high = mid - 1
    return points[:k]
#Partition, Time worst O(n^2) average O(n), Space O(1)
def partition(points, low, high) :
    pivot = points[low]
    while (low < high) :
        while low < high and compare(points[high], pivot) >= 0: 
            high -= 1
        points[low] = points[high]
        while low < high and compare(points[low], pivot) <= 0: 
            low += 1
        points[high] = points[low]
    points[low] = pivot
    return low
#Compare based on Euclidean distance, Time O(1), Space O(1)
def compare(a, b) :
    return (a[0]*a[0] + a[1]*a[1]) - (b[0]*b[0] + b[1]*b[1])
# kmeans optimization
def assignment_step_v5(data, centroids):
    diff = data[:, None] - centroids[None]  # (n, k, d)
    distances = np.einsum('nkd,nkd->nk', diff, diff)  # (n, k)
    labels = np.argmin(distances, axis=1)  # (n,)
    return labels

#%% Deals with multuple templates
class Matching3D:
    
    def __init__(self, config = None):
               
        self.state              = False  # object is initialized
        self.cfg                = config
        self.debugOn            = True
        self.figNum             = 1      # control names of the figures
        
        self.srcObj3d           = None   # model to be matched
        self.dstObj3d           = None   # target to be matched to
        
        self.srcDist            = None   # distance hash for source
        self.dstDist            = None   # distance hash for target
        
        self.srcPairs           = None   # pairs hash for source (inverse index of srcDist)
        self.dstPairs           = None   # pairs hash for target (inverse index of dstDist)
        
        self.srcAdjacency       = None   # adjacency mtrx for source
        self.dstAdjacency       = None   # adjacency mtrx for target 
        
        self.srcC               = None   # center vector 
        self.srcE               = None   # extension vector 
        self.dstC               = None   # center vector 
        self.dstE               = None   # extension vector        
               
        self.Print('3D-Manager is created')
        
    def SelectTestCase(self, testType = 1):        
        # loads 3D data by test specified
        ret = False
        if testType == 1:    
            point_data  = get_test_vertex(1)*10
            source      = get_pcd_from_vertex(point_data)
            source.translate((1, 1, 1))
            source.paint_uniform_color([0.1, 0.8, 0.1])
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1])
     
        elif testType == 2:
            point_data  = get_test_vertex(10)
            source      = get_pcd_from_vertex(point_data)
            source.translate((1, 1, 1))
            source.paint_uniform_color([0.1, 0.8, 0.1])
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1])
            
        elif testType == 3:
            point_data  = get_test_vertex(11)
            source      = get_pcd_from_vertex(point_data)
            source.translate((1, 1, 1))
            source.paint_uniform_color([0.1, 0.8, 0.1])
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1])
            
        elif testType == 4:
            self.Print('two point clouds with different extensions')
            point_data  = get_test_vertex(1)
            source      = get_pcd_from_vertex(point_data)
            source.paint_uniform_color([0.1, 0.8, 0.1])
            source.scale(2, (1, 1, 1))
            point_data  = get_test_vertex(2)
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1])  
            
        elif testType == 11:
            self.Print("one point cloud is a part of the other.")
            point_data  = get_test_vertex(11)
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1]) 
            point_data  = point_data[:27,:]
            source      = get_pcd_from_vertex(point_data)
            source.paint_uniform_color([0.1, 0.8, 0.1])
            source.translate((1, 1, 1))
            
        elif testType == 12:
            self.Print("one point cloud is a part of the other with noise.")
            point_data  = get_test_vertex(12)
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1]) 
            point_data  = point_data[:7,:]
            source      = get_pcd_from_vertex(point_data)
            #source      = apply_noise(source, 0, 0.01)
            source.paint_uniform_color([0.1, 0.8, 0.1]) 
            source.translate((1, 1, 1))          

        elif testType == 21:    
            pcd_data = o3d.data.DemoICPPointClouds()
            source = o3d.io.read_point_cloud(pcd_data.paths[0])
            target = o3d.io.read_point_cloud(pcd_data.paths[1])
     
        elif testType == 22:
            print("Load two aligned point clouds.")
            demo_data = o3d.data.DemoFeatureMatchingPointClouds()
            source = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
            target = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])
                                              
        elif testType == 31:
            self.Print("Load customer models.")
            dataPath = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\valve.laz')
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            source          = o3d.geometry.PointCloud()
            point_data      = point_data / 1e6 # rescale from um to m
            source.points   = o3d.utility.Vector3dVector(point_data)
            source.paint_uniform_color([0.5, 0.3, 0.1])
            source.translate((1, 1, 1))
            las             = laspy.read(dataPath + '\\valve.laz') # valve pressure # farm2M
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            target          = o3d.geometry.PointCloud()
            point_data      = point_data / 1e6 # rescale from um to m
            target.points   = o3d.utility.Vector3dVector(point_data)
            target.paint_uniform_color([0.1, 0.3, 0.5])
            
                        
        else:
            self.Print('Bad choice %d is selected' %testType, 'E')
            return ret    
        
        # output
        self.srcObj3d           = source   # model to be matched
        self.dstObj3d           = target   # target to be matched to

        self.Print('Data set %d is selected' %testType, 'I')
        return True  
        
    def DataStatistics(self, pcd):
        # computes statistics
        ret = False
        axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
        axis_aligned_bounding_box.color = (1, 0, 0)
        v_ext = axis_aligned_bounding_box.get_extent()
        print('Extension axis : %s' %str(v_ext))
        v_center = axis_aligned_bounding_box.get_center()
        print('Center coordinates : %s' %str(v_center))
        
        return v_center,  v_ext    
    
    def Downsample(self, pcd):
        # reduce num of points
        #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=100.5)
        sample_rate = int(len(pcd.points)/1000)
        if sample_rate > 1:
            down_pcd = pcd.uniform_down_sample(sample_rate)
            self.Print('Downsampling data by factor %d' %sample_rate)
        else:
            down_pcd = pcd
            self.Print("No Downsample of the point cloud")
            
        #o3d.visualization.draw([down_pcd])
        return down_pcd
    
    def RemoveOutliers(self, pcd):
        # remove outlier points        
        print("Statistical oulier removal")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])
        
        return True
     
    def PrepareDataset(self, pcd, min_dist_value = 0.0):
        # prepares edge hash and also reverse 
        t_start      = time.time()
        #edgeDict, distDict = distance_hash_python_fast(pcd.points)
        edgeDict, distDict = distance_hash_python_knn(pcd.points, min_dist_value = min_dist_value, knn_num = 11)

        t_stop      = time.time()
        print('Dist Hash time : %4.3f [sec]' %(t_stop - t_start))

        # extract pairs and build adjacency
        #point_num = len(pcd.points)
        #pairs    = distDict.keys()
        adjMtrx  = None #adjacency_matrix(pairs, point_num)
        
        return edgeDict, distDict, adjMtrx
                   
    def Preprocess(self):
        # computes hash functions for source and target
        self.dstObj3d = self.Downsample(self.dstObj3d)
        self.srcObj3d = self.Downsample(self.srcObj3d)
        
        # compute different structures
        self.dstDist, self.dstPairs, self.dstAdjacency = self.PrepareDataset(self.dstObj3d)
        self.srcDist, self.srcPairs, self.srcAdjacency = self.PrepareDataset(self.srcObj3d)
        
        # center and dimensions
        self.srcC, self.srcE = self.DataStatistics(self.srcObj3d)
        self.dstC, self.dstE = self.DataStatistics(self.dstObj3d)        
    
        return True
    
    def PreprocessSingle(self,pcd, factor = 10):
        # computes hash functions for source and target
        dstPcd = self.Downsample(pcd)
        
        # center and dimensions
        dstC, dstE = self.DataStatistics(dstPcd)    
        
        # minimal distance between points
        min_dist   = (dstE.max()/factor)**2   
        
        # compute different structures
        dstDist, dstPairs, dstAdjacency = self.PrepareDataset(dstPcd, min_dist)

        return dstPcd, dstDist, dstPairs, dstC, dstE
    
    def MatchCycle3(self, indx = []):
        # match in several steps
        #if self.dstAdjacency is None:
        #    self.Preprocess()

        if len(indx) < 3:
            # extract distances for the designated points
            snodes      = [1,2,3]
        else:
            snodes      = indx
            
        sshift      = np.roll(snodes,-1) #snodes[1:] + snodes[:1]  # shift nodes by 1
        spairs      = [(snodes[k],sshift[k]) for k in range(len(snodes))]

        # move over from point to point and store all possible matches
        t_start     = time.time()
        cycle_list  = []
        count_cycle = 0
        for sjk in spairs:
            
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
            print("set dij:\n",dpairs_ij)
        
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
        print('Match time : %4.3f [sec]' %(t_stop - t_start))
        
        self.srcObj3d = self.ColorCyles(self.srcObj3d, spairs_ii, [1,0,0])
        self.dstObj3d = self.ColorCyles(self.dstObj3d, dpairs_ii, [0,1,0])
            
        return True
        
    def ColorSpecificPoints(self, pcd, pairs_ij, clr = [0, 1, 0]):   
        self.Print('Colors specific points and the rest are gray') 
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if pairs_ij is None:
            return pcd
        
        idx = [s[0] for s in pairs_ij]
        np.asarray(pcd.colors)[idx, :] = clr
        
        return pcd
    
    def ColorCyles(self, pcd, cycle_ii, clr = [0, 1, 0]):   
        # cycle_ii is a dictionary with cycle per point
        #self.Print('Colors point cycles and the rest are gray') 
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        if cycle_ii is None:
            return pcd
        
        for p in cycle_ii.keys():
            idx = cycle_ii[p]
            np.asarray(pcd.colors)[idx, :] = clr
        
        return pcd
        
    def ShowData3D(self, src, dst, transformation = None, wname = 'Source'):
        # show 3D data
        ret = False
        # if not isinstance(src, type(o3d)):
        #     self.Print('Need open3d object - src')
        #     return ret
        # if not isinstance(dst, type(o3d)):
        #     self.Print('Need open3d object - dst')
        #     return ret
        
        if transformation is None:
            transformation = np.eye(4)      
        
        source_temp = copy.deepcopy(src)
        target_temp = copy.deepcopy(dst)
        #source_temp.paint_uniform_color([0.9, 0.1, 0])
        #target_temp.paint_uniform_color([0, 0.9, 0.1])
        #source_temp.transform(transformation)
        o3d.visualization.draw([source_temp, target_temp])
        #o3d.visualization.draw_geometries([source_temp, target_temp], window_name = wname)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                               zoom=0.4559,
        #                               front=[0.6452, -0.3036, -0.7011],
        #                               lookat=[1.9892, 2.0208, 1.8945],
        #                               up=[-0.2779, -0.9482, 0.1556])        
        return True
    
        # point-to-point ICP for refinement
        self.Print("Perform point-to-point ICP refinement")
        
        source, target, trans_init = self.srcObj3d, self.dstObj3d, np.eye(4)
        
        threshold = 0.03  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # debug
        #self.ShowData3D(source, target, reg_p2p.transformation)
        print(reg_p2p.transformation)
        return reg_p2p.transformation

    def Finish(self):
        
        #cv.destroyAllWindows() 
        self.Print('3D-Manager is closed')
        
    def Print(self, txt='',level='I'):
        print('%s: 3DM: %s' %(level, txt))

        
       
#%% --------------------------           
class TestMatching3D(unittest.TestCase):                
    def test_Create(self):
        d       = Matching3D()
        self.assertEqual(False, d.state)
        
    def test_LoadData(self):
        # check read training files  
        d           = Matching3D()
        isOk        = d.SelectTestCase(1)
        self.assertTrue(isOk)
        
    def test_ShowData(self):
        # check data show  
        d           = Matching3D()
        isOk        = d.SelectTestCase(3)
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        #isOk        = d.SelectTestCase(2)
        #d.ShowData3D(d.srcObj3d,d.dstObj3d, wname = 'Case 2')
        self.assertTrue(isOk)  
        
    def test_Transform(self):
        # test trasnformation  
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
        transform   = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])
        d.ShowData3D(d.srcObj3d,d.srcObj3d, transform)
        self.assertTrue(isOk)   
        
    def test_RemoveOutliers(self):
        # test outlier point removal - ok  
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
        isOk        = d.RemoveOutliers(d.srcObj3d)
        self.assertTrue(isOk)  
        
    def test_Downsample(self):
        # test trasnformation  
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
        srcDown     = d.Downsample(d.srcObj3d)
        transform   = np.array([[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]])
        d.ShowData3D(d.srcObj3d,srcDown, transform)
        self.assertTrue(isOk)  
        
    def test_Statistics(self):
        # statistics about the data  
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
    
        d.DataStatistics(d.srcObj3d)
        d.DataStatistics(d.dstObj3d)
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        self.assertTrue(isOk)                             
        
    def test_MatchICP(self):
        # check ICP
        d           = Matching3D()
        isOk        = d.SelectTestCase(4)
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        transformation = d.MatchP2P()
        d.ShowData3D(d.srcObj3d,d.dstObj3d, transformation)
        self.assertTrue(isOk)  
        
    def test_MatchPairs(self):
        # check Hash matching for a single pair
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
        
        d.MakeHashAndMatch()
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        self.assertTrue(isOk)    
        
    def test_PreprocessAndMatch(self):
        # match cycle 
        d           = Matching3D()
        isOk        = d.SelectTestCase(1)
        
        d.PreprocessAndMatch()
        #d.ShowData3D(d.srcObj3d,d.dstObj3d)
        self.assertTrue(isOk)                        
        
    def test_FindClosestPoints(self):
        # find KNN like points with distance  
        d           = Matching3D()
        # test case definition
        points      = get_test_vertex(11)
        pcd         = get_pcd_from_vertex(points)
        pcd_knn, _  = find_closest_points(pcd, point_coord = [0,0,0], min_dist = 85, max_dist = 175, point_num = 5)

        d.ShowData3D(pcd, pcd_knn)
        self.assertTrue(len(pcd_knn.points) < 10)  
        
    def test_MatchCycle3(self):
        # match cycle 
        d           = Matching3D()
        isOk        = d.SelectTestCase(11)
    
        # preprocess all the dta
        d.dstObj3d, d.dstDist, d.dstPairs, _ , _                = d.PreprocessSingle(d.dstObj3d, 1000)
        # factor 1000 make min distance small and keeps all the points
        d.srcObj3d, d.srcDist, d.srcPairs, d.srcC, d.srcE       = d.PreprocessSingle(d.srcObj3d, 1000) 
         

        # find the closest points for selection
        min_dist   = (d.srcE.max()/100)**2
        max_dist   = (d.srcE.max()/1)**2
        pcd_knn, indx  = find_closest_points(d.srcObj3d, point_coord = d.srcC, min_dist = min_dist, max_dist = max_dist, point_num = 3)
        
        #indx        = indx[1,2,3,4]
        isOk        = d.MatchCycle3(indx)

        d.ShowData3D(d.srcObj3d,  d.dstObj3d)
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
        import numpy as np
        
        from sklearn.neighbors import KDTree

        X = np.random_sample((10, 3))  # 10 points in 3 dimensions
        tree = KDTree(X, leaf_size=2)              
        dist, ind = tree.query(X[:1], k=3)                
        print(ind)  # indices of 3 closest neighbors

        print(dist)  # distances to 3 closest neighbors


    
               

#%%
if __name__ == '__main__':
    #print (__doc__)
    
    #unittest.main()
    
    # template manager test
    singletest = unittest.TestSuite()
#    singletest.addTest(TestMatching3D("test_Create"))
#    singletest.addTest(TestMatching3D("test_LoadData"))
#    singletest.addTest(TestMatching3D("test_ShowData"))  
#    singletest.addTest(TestMatching3D("test_Transform"))
#    singletest.addTest(TestMatching3D("test_RemoveOutliers"))
#    singletest.addTest(TestMatching3D("test_Downsample"))
#    singletest.addTest(TestMatching3D("test_Statistics"))  
#    singletest.addTest(TestMatching3D("test_MatchPairs"))
#    singletest.addTest(TestMatching3D("test_PreprocessAndMatch"))
    singletest.addTest(TestMatching3D("test_MatchCycle3"))
#    singletest.addTest(TestMatching3D("test_FindClosestPoints"))  
#    singletest.addTest(TestMatching3D("test_Distance"))
    
    unittest.TextTestRunner().run(singletest)