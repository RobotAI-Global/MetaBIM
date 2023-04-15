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
       

-----------------------------
 Ver	Date	 Who	Descr
-----------------------------
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


SENSOR_NOISE       = 1      # sensor noise in mm
MAX_OBJECT_SIZE    = 1000   # max object size to be macthed 

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

def get_pcd_from_vertex(point_data, clr = [0.3, 0.3, 0.3] ):
    # transform points to pcd

    pcd             = o3d.geometry.PointCloud()
    pcd.points      = o3d.utility.Vector3dVector(point_data)
    pcd.paint_uniform_color(clr)
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


#%% Deals with multuple templates
class MatchPointClouds:
    
    def __init__(self, config = None):
               
        self.state              = False  # object is initialized
        self.cfg                = config
        self.debugOn            = True
        self.figNum             = 1      # control names of the figures
        self.t_start            = time.time()
        
        # param
        # DIST_BIN_WIDTH helps to assign multiple distances to the same bin. 
        # higher DIST_BIN_WIDTH more distances in the same bin
        self.DIST_BIN_WIDTH   = 10     # how many right zeros in the distance number
        
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
        
        # self.srcC               = None   # center vector 
        # self.srcE               = None   # extension vector 
        # self.dstC               = None   # center vector 
        # self.dstE               = None   # extension vector        
               
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
            self.Print("one point cloud is a part of the other with noise.")
            point_data  = get_test_vertex(12)
            self.dst_points = point_data
            point_data  = point_data[:7,:]
            self.src_points = point_data + 1  # small shift to see the difference        

                                              
        elif testType == 31:
            self.Print("Load customer models.")
            dataPath = r'C:\RobotAI\Customers\MetaBIM\Code\MetaBIM\Data\2023-02-10'
            las             = laspy.read(dataPath + '\\valve.laz')
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.dst_points = point_data
            las             = laspy.read(dataPath + '\\valve.laz') # valve pressure # farm2M
            point_data      = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            self.src_points = point_data + 1  # small shift to see the difference 
            
                        
        else:
            self.Print('Bad choice %d is selected' %testType, 'E')
            return ret    
        
        # output
        self.Print('Data set %d is selected' %testType, 'I')
        return True    
    
    def Downsample(self, points):
        # reduce num of points
        #voxel_down_pcd = pcd.voxel_down_sample(voxel_size=100.5)
        sample_rate = int(len(points)/1000)
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

    def PrepareDataset(self, points, min_dist_value = 0, max_dist_value = MAX_OBJECT_SIZE):
        # compute distances only on the part
        dist_ij     = scipy_cdist(points,points)
        # dist_ij[dist_ij < min_dist_value] = 1e9
        # dkey_ij     = dist2key(dist_ij,10)
        # index_ij    = dkey_ij.argsort(axis = 1)
        # knn_num     = np.minimum(knn_num,len(points))
        
        dist_ij       = dist2key(dist_ij, self.DIST_BIN_WIDTH)
        index_i, index_j      = np.nonzero(np.logical_and(min_dist_value < dist_ij , dist_ij < max_dist_value))
    
        dist_dict = {}
        pairs_dict = {}
        for i,j in zip(index_i, index_j):

            add_value(dist_dict, dist_ij[i,j], (i,j))
            add_value(pairs_dict,(i,j), dist_ij[i,j])

        return dist_dict, pairs_dict

    def PreprocessPointData(self, points, factor = 10):
        # computes hash functions for source and target
        points          = self.Downsample(points)
        
        # make it compact
        points          = self.MakeCompact(points)
        
        # minimal distance between points
        min_dist            = SENSOR_NOISE * 10   
        
        # compute different structures from distance
        t_start      = time.time()
        dist_dict, pairs_dict   = self.PrepareDataset(points,  min_dist, MAX_OBJECT_SIZE)
        t_stop      = time.time()
        self.Print('Dist Hash time : %4.3f [sec]' %(t_stop - t_start))
        
        return points,  dist_dict, pairs_dict
    
    def SelectMatchPoints(self, points, selectType = 1):
        # select the points according to different criteria   
        point_num = points.shape[0]
        
        if selectType == 1:
            # definien
            indx = [1,2,3]
           
        elif selectType == 2:
             # random
            indx = np.random.randint(0, point_num, size=3)
                  
        elif selectType == 11:
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
            
        self.src_cycle         = spairs_ii   # cycle selected
        self.dst_cycle         = dpairs_ii   # cycles detected     
            
        return True
        
    def MatchSourceToTarget(self, src_points = None, dst_points = None):
        # main processing and matching
        if src_points is None or dst_points is None:
            src_points, dst_points = self.src_points, self.dst_points
        
        # preprocess all the dta
        src_points, src_dist_dict, src_pairs_dict  = self.PreprocessPointData(src_points)
        
        # factor 1000 make min distance small and keeps all the points
        dst_points, dst_dist_dict, dst_pairs_dict  = self.PreprocessPointData(dst_points)
         
        # select indexes
        src_indx = self.SelectMatchPoints(src_points, selectType = 1)

        #indx        = indx[1,2,3,4]
        self.srcPairs   = src_pairs_dict
        self.dstDist    = dst_dist_dict
        isOk            = self.MatchCycle3(src_indx) 

        
        return
        
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
        src_pcd = get_pcd_from_vertex(self.src_points, clr = [0.8, 0.4, 0.1] )
        dst_pcd = get_pcd_from_vertex(self.dst_points, clr = [0.4, 0.8, 0.1] )
       
        # color cycles
        src_pcd = self.ColorCyles(src_pcd, self.src_cycle, [1,0,0])
        dst_pcd = self.ColorCyles(dst_pcd, self.dst_cycle, [0,1,0])    


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
        return True


    def Finish(self):
        
        #cv.destroyAllWindows() 
        self.Print('3D-Manager is closed')
        
    def Print(self, txt='',level='I'):
        t_stop = time.time()
        print('%s: 3DM: %4.3f [s]: %s' %(level, t_stop-self.t_start, txt))
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
                                    
        
    def test_MatchSourceTarget(self):
        # match cycle 
        d           = MatchPointClouds()
        isOk        = d.SelectTestCase(1)
    
        isOk        = d.MatchSourceToTarget()

        d.ShowData3D()
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
#    singletest.addTest(TestMatchPointClouds("test_ShowData"))  
#    singletest.addTest(TestMatching3D("test_Distance"))
    singletest.addTest(TestMatchPointClouds("test_MatchSourceTarget"))

    
    unittest.TextTestRunner().run(singletest)