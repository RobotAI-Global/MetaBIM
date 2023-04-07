'''
3D Data Matching

Matches 3D datasets using xgboost

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
        point_data = np.random.rand(10,3)*100        

    elif vtype == 11: # random with 1 match
        point_data = np.random.rand(256,3)*100

    elif vtype == 12: # random with 2 matches
        point_data = np.random.rand(10,3)*10
        point_data = np.vstack((point_data,point_data[::-1,:]+10))
    
    else:
        ValueError('bad vtype')
        
    return point_data

def get_pcd_from_vertex(point_data):
    # transform points to pcd

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)
    #pcd.paint_uniform_color([0.3, 0.3, 0.3])
    #o3d.visualization.draw([pcd])
    return pcd

def apply_noise(pcd, mu = 0, sigma = 1):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

# ======================
def dist2key(d):
    # creates key from distance
    k = np.round(d*10)/10
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

def radius_search(pcd, point_coord = [0,0,0], search_dist = 0.2):
    # search by radius
    print("Find the neighbors of %s point with distance less than %d"  %(str(point_coord),search_dist ))
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(point_coord, search_dist)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

    print("Displaying the point cloud ...\n")
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
            point_data  = get_test_vertex(10)
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1]) 
            point_data  = point_data[:7,:]
            source      = get_pcd_from_vertex(point_data)
            source.paint_uniform_color([0.1, 0.8, 0.1])
            source.translate((1, 1, 1))
            
        elif testType == 12:
            self.Print("one point cloud is a part of the other with noise.")
            point_data  = get_test_vertex(11)
            target      = get_pcd_from_vertex(point_data)
            target.paint_uniform_color([0.8, 0.1, 0.1]) 
            point_data  = point_data[:7,:]
            source      = get_pcd_from_vertex(point_data)
            source      = apply_noise(source, 0, 0.1)
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
            las = laspy.read(dataPath + '\\valve.laz')
            point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(point_data)
            source.translate((1, 1, 1))
            las = laspy.read(dataPath + '\\valve.laz') # valve pressure # farm2M
            point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(point_data)
            
                        
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
        
        return ret      
    
    def Downsample(self, pcd):
        # reduce num of points
        print("Downsample the point cloud with a voxel of 0.02")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=100.5)
        o3d.visualization.draw([voxel_down_pcd])
        return voxel_down_pcd
    
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
     
    def PrepareDataset(self, pcd):
        # prepares edge hash and also reverse 
        edgeDict, distDict = distance_hash_python(pcd.points)
        
        # extract pairs and build adjacency
        point_num = len(pcd.points)
        pairs    = distDict.keys()
        adjMtrx  = adjacency_matrix(pairs, point_num)
        
        return edgeDict, distDict, adjMtrx
            
     
    def MakeHashAndMatch(self):
        # computes hash functions for source and target
        self.dstDist = distance_hash_python(self.dstObj3d.points)
        self.srcDist = distance_hash_python(self.srcObj3d.points)
        
        # assume that src is smaller
        dist_points = list(self.srcDist.keys())[1]
        sij     = self.srcDist[dist_points]
           
        dij     = get_match_pairs(self.dstDist, dist_points)
        #if isinstance(pairs_jk_est[k],list):
        #print()
        print("set sij:",sij)
        print("set dij:",dij)
        
        self.srcObj3d = self.ColorSpecificPoints(self.srcObj3d, sij, [1,0,0])
        self.dstObj3d = self.ColorSpecificPoints(self.dstObj3d, dij, [0,1,0])
            
        return 
        
    def PreprocessAndMatch(self):
        # computes hash functions for source and target
        self.dstDist, self.dstPairs, self.dstAdjacency = self.PrepareDataset(self.dstObj3d)
        self.srcDist, self.srcPairs, self.srcAdjacency = self.PrepareDataset(self.srcObj3d)
        
        # extract distances for the designated points
        snodes      = [1,2,3]
        sshift      = np.roll(snodes,-1) #snodes[1:] + snodes[:1]  # shift nodes by 1
        spairs      = [(snodes[k],sshift[k]) for k in range(3)]
        
        # matching the first cycle
        match_dict = {}
        count_dict = {}
        pairs_dict = {}
        count_cycle = 0
        for sij in spairs:
            sdist        = self.srcPairs[sij]  # extract distance
            dij_list     = get_match_pairs(self.dstDist, sdist)
            for (i,j) in dij_list:
                if count_cycle == 0:
                    count_dict[i] = count_cycle  # starting point
                    
                # if previous entry exists - could be a cycle
                cnt = count_dict.get(i)      
                if cnt is None:
                    continue
                # if it equals to the cycle count - add to the list
                if cnt == (count_cycle - 1):           
                    add_value(match_dict, i, j) # list of connections                   
                    count_dict[j] = count_cycle + 1 # next point

            count_cycle += 1 

            print("set sij:",sij)
            print("set dij:",dij_list)
        
        #self.srcObj3d = self.ColorSpecificPoints(self.srcObj3d, sij, [1,0,0])
        #self.dstObj3d = self.ColorSpecificPoints(self.dstObj3d, dij, [0,1,0])
            
        return 
        
    def MatchCycle3(self):
        # match in several steps
        self.dstDist, self.dstPairs, self.dstAdjacency = self.PrepareDataset(self.dstObj3d)
        self.srcDist, self.srcPairs, self.srcAdjacency = self.PrepareDataset(self.srcObj3d)
        
        # extract distances for the designated points
        snodes      = [1,2,3]
        sshift      = np.roll(snodes,-1) #snodes[1:] + snodes[:1]  # shift nodes by 1
        spairs      = [(snodes[k],sshift[k]) for k in range(3)]

        # move over from point to point and store all possible matches
        cycle_list  = []
        count_cycle = 0
        for sjk in spairs:
            
            dist_jk        = self.srcPairs[sjk]  # extract distance
            dpairs_jk      = get_match_pairs(self.dstDist, dist_jk)
            # init ij - first point
            if count_cycle == 0:
                # use dictionary to trace cycles
                dpairs_ij = {djk:[djk[0]] for djk in dpairs_jk}  # starting point
            else:
                dpairs_ij = dict_intersect(dpairs_ij, dpairs_jk)
            # store pairs for traceback
            cycle_list.append(dpairs_ij)
            count_cycle += 1 
            print("set dij:",dpairs_ij)
            
        # extract cycles
        dpairs_ii = cycle_list[-1]
        spairs_ii = {(snodes[0],snodes[0]) : snodes}
        print("scycle dii:",spairs)
        print("dcycle dii:",dpairs_ii)
        
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
        self.Print('Colors point cycles and the rest are gray') 
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
        
    def test_MatchCycle3(self):
        # match cycle 
        d           = Matching3D()
        isOk        = d.SelectTestCase(3)
        
        t_start     = time.time()
        isOk        = d.MatchCycle3()
        t_stop      = time.time()
        print('Match time : %4.3f [sec]' %(t_stop - t_start))
        
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        self.assertTrue(isOk) 

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
  
    
    unittest.TextTestRunner().run(singletest)