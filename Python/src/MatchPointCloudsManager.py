'''
3D Point Cloud Data Manager

Test set manager.

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
0301   28.09.23  UD     removing open3d connection
0202   28.04.23  UD     New data for match
0201   14.04.23  UD     simplifying the Matching3D
0102   28.02.23  UD     adding laz file support
0101   25.02.23  UD     Created
-----------------------------

'''

import os
import numpy as np
#import open3d as o3d
import laspy
import time
#import cv2 as cv
#import json
#import copy
from scipy.spatial.distance import cdist, pdist, squareform
import unittest
import matplotlib.pyplot as plt

plt.ioff()  # interactive mode off

#%% Help functions



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

#%% Deals with multuple templates
class MatchPointCloudsManager:
    
    def __init__(self, config = None, is_source = True, points_data = None):
               
        self.cfg                = config
        self.debugOn            = True
        self.figNum             = 1      # control names of the figures
        self.t_start            = time.time()
        
        if is_source:
            self.color              = (1,0,0)
            self.type               = 'SRC'
        else:
            self.color              = (0,1,0)
            self.type               = 'DST'        
        
        self.SENSOR_NOISE       = 1      # sensor noise in mm
        self.MAX_OBJECT_SIZE    = 3000   # max object size to be macthed         
        self.DOWNSAMPLE         = 1     # downsample the sourcs model
        
        # param
        # DIST_BIN_WIDTH helps to assign multiple distances to the same bin. 
        # higher DIST_BIN_WIDTH more distances in the same bin
        self.DIST_BIN_WIDTH     = 1     # how many right zeros in the distance number
              
        self.points             = points_data   # model points Nx3 to be matched
         
        self.cycle              = None   # cycle selected
                     
        self.obj3d              = None   # model to be matched
        
        self.dist_dict          = None   # distance hash for source
        
        self.pairs_dict         = None   # pairs hash for source (inverse index of dist_hash)

        self.dbg_dist_value     = None    # jusy for debug
        
        # gui
        self.gui_handles        = {'f':None, 'a': None, 'p': None} 
   
               
        self.Print('3D-Manager is created')
         
    
    def Downsample(self, points, factor = 1):
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

    def PreprocessPointData(self, points = None, factor = 1):
        # computes hash functions for source or target
        
        points          = self.points if points is None else points
        
        # make smaller
        points          = self.Downsample(points, factor)
        
        # make it compact
        points          = self.MakeCompact(points)
        
        # minimal distance between points
        min_dist        = self.SENSOR_NOISE * 10   
        max_dist        = self.MAX_OBJECT_SIZE
        
        # compute different structures from distance
        t_start         = time.time()
        dist_dict, pairs_dict   = self.PrepareDataset(points,  min_dist, max_dist)
        t_stop          = time.time()
        self.Print('Dist Hash time : %4.3f [sec]' %(t_stop - t_start))
        
        
        self.points         = points
        self.dist_dict      = dist_dict
        self.pairs_dict     = pairs_dict
        return points,  dist_dict, pairs_dict
    
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
    
    def ShowData3D(self, points = None):
        # show point cloud
        if points is None:
            points = self.points
            
        fig_num = 1 if self.type == 'SRC' else 2

        
     
        #fig = plt.figure(fig_num)
        #plt.ion() 
        #fig.canvas.set_window_title('3D Scene')
        #ax = fig.gca() #projection='3d')
        #
          
        min_values1, max_values1   = points.min(axis=0), points.max(axis=0)    
        X_min, Y_min, Z_min        = min_values1
        X_max, Y_max, Z_max        = max_values1
        
        max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 1.9 # to get some volume 2.0
    
        mid_x = (X_max+X_min) * 0.5
        mid_y = (Y_max+Y_min) * 0.5
        mid_z = (Z_max+Z_min) * 0.5
        
        # show
        fig         = plt.figure(fig_num)
        ax          = fig.add_subplot(projection='3d')
        p           = ax.scatter3D(points[:,0],points[:,1],points[:,2], color=self.color, s=16.0)
        
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  
        ax.set_aspect("equal")
        
        plt.title('%s 3D Point Data' %self.type)
        plt.show(block=False)
        self.Print('Cloud rendering is done')
        
        self.gui_handles = {'f':fig, 'a': ax, 'p': p}        
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
        print('%s: %s: %4.3f: %s' %(level, self.type, t_stop-self.t_start, txt))
        self.t_start = time.time()    
       
#%% --------------------------           
from MatchPointCloudsDatasets import MatchPointCloudsDatasets

class TestMatchPointCloudsManager(unittest.TestCase):                
    def test_Create(self):
        d       = MatchPointCloudsManager(None, is_source = False)
        self.assertEqual('DST', d.type)
        
    def test_LoadData(self):
        # check read data file and init
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(1)
        m           = MatchPointCloudsManager(None, True, d.src_points)                
        self.assertTrue(len(m.points) > 0)
        
    def test_ShowData(self):
        # check data show  
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(3)
        m           = MatchPointCloudsManager(None, True, d.src_points) 
        m.ShowData3D()
        self.assertTrue(isOk) 
        
    def test_ShowDataTwoObjects(self):
        # check data show  
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(11)
        ms          = MatchPointCloudsManager(None, True, d.src_points) 
        md          = MatchPointCloudsManager(None, False, d.dst_points)
        ms.ShowData3D()
        md.ShowData3D()
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
        self.assertTrue(len(sp_result) == 10000)
            
    def test_Downsample(self):
        # test downsample  
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(3)
        ms          = MatchPointCloudsManager(None, True, d.src_points) 
        ms.points   = ms.Downsample(ms.points)
        ms.ShowData3D()
        self.assertTrue(isOk)  
        
    def test_PreprocessPointData(self):
        # test preprocess  
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(3)
        ms          = MatchPointCloudsManager(None, True, d.src_points) 
        res         = ms.PreprocessPointData(ms.points)
        ms.ShowData3D()
        self.assertTrue(isOk)          
        
        
    
    def test_Histogram(self):
        # test distance histograms
        d           = MatchPointCloudsDatasets()
        isOk        = d.SelectTestCase(3)  # 62-ok 
        ms          = MatchPointCloudsManager(None, True,  d.src_points) 
        md          = MatchPointCloudsManager(None, False, d.dst_points)        
        md.Print('Target model...')
        dst_points, dst_dist_dict, dst_pairs_dict  = md.PreprocessPointData(md.points, md.DOWNSAMPLE)     
        md.ShowDistanceHistogram(md.dbg_dist_value,'Dst Distances')
        md.ShowDictionaryHistogram(dst_dist_dict,'Dst Dictionary')
        
        # preprocess all the dta
        ms.Print('Match model...')
        src_points, src_dist_dict, src_pairs_dict  = ms.PreprocessPointData(ms.points, ms.DOWNSAMPLE)  
        ms.ShowDistanceHistogram(ms.dbg_dist_value, 'Src Distances')
        ms.ShowDictionaryHistogram(src_dist_dict,'Src Dictionary')
        
       
        
#%%
if __name__ == '__main__':
    #print (__doc__)
    
    #unittest.main()
    
    # template manager test
    singletest = unittest.TestSuite()
    #singletest.addTest(TestMatchPointCloudsManager("test_Create")) #ok
    #singletest.addTest(TestMatchPointCloudsManager("test_LoadData")) #ok
    #singletest.addTest(TestMatchPointCloudsManager("test_ShowData")) # ok 
    #singletest.addTest(TestMatchPointCloudsManager("test_ShowDataTwoObjects")) # ok
      
    #singletest.addTest(TestMatchPointCloudsManager("test_Distance"))  # ok
    #singletest.addTest(TestMatchPointCloudsManager("test_Downsample")) #ok
    singletest.addTest(TestMatchPointCloudsManager("test_PreprocessPointData")) #ok
   #singletest.addTest(TestMatchPointCloudsManager("test_Histogram"))
    
    unittest.TextTestRunner().run(singletest)