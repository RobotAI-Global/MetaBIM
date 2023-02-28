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




#%% Deals with multuple templates
class Matching3D:
    
    def __init__(self, config = None):
               
        self.state              = False  # object is initialized
        self.cfg                = config
        self.debugOn            = True
        self.figNum             = 1      # control names of the figures
        
        self.srcObj3d           = None   # model to be matched
        self.dstObj3d           = None   # target to be matched to
               
        self.Print('3D-Manager is created')
        
    def SelectTestCase(self, testType = 1):        
        # loads 3D data by test specified
        ret = False
        if testType == 1:    
            pcd_data = o3d.data.DemoICPPointClouds()
            source = o3d.io.read_point_cloud(pcd_data.paths[0])
            target = o3d.io.read_point_cloud(pcd_data.paths[1])
     
        elif testType == 2:
            print("Load two aligned point clouds.")
            demo_data = o3d.data.DemoFeatureMatchingPointClouds()
            source = o3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
            target = o3d.io.read_point_cloud(demo_data.point_cloud_paths[1])
            
        elif testType == 11:
            print("Load customer models.")
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
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw([source_temp, target_temp])
        #o3d.visualization.draw_geometries([source_temp, target_temp], window_name = wname)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                               zoom=0.4559,
        #                               front=[0.6452, -0.3036, -0.7011],
        #                               lookat=[1.9892, 2.0208, 1.8945],
        #                               up=[-0.2779, -0.9482, 0.1556])        
        return True
    
    def ComputeKeypoints(self,pcd):
        # compute key points of the pcloud
        tic = time.time()
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
        toc = 1000 * (time.time() - tic)
        print("ISS Computation took {:.0f} [ms]".format(toc))

        #mesh.compute_vertex_normals()
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        keypoints.paint_uniform_color([1.0, 0.75, 0.0])
        o3d.visualization.draw_geometries([keypoints, pcd])
        return keypoints    
        
    def MatchP2P(self):
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
        
        if level == 'I':
            ptxt = 'I: 3DM: %s' % txt
            #logging.info(ptxt)  
        if level == 'W':
            ptxt = 'W: 3DM: %s' % txt
            #logging.warning(ptxt)  
        if level == 'E':
            ptxt = 'E: 3DM: %s' % txt
            #logging.error(ptxt)  
           
        print(ptxt)
        
       
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
        isOk        = d.SelectTestCase(11)
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        #isOk        = d.SelectTestCase(2)
        #d.ShowData3D(d.srcObj3d,d.dstObj3d, wname = 'Case 2')
        self.assertTrue(isOk)  
        
    def test_Transform(self):
        # test trasnformation  
        d           = Matching3D()
        isOk        = d.SelectTestCase(1)
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
        
    def test_MatchP2P(self):
        # check ICP P2P matching  
        d           = Matching3D()
        isOk        = d.SelectTestCase(1)
        d.ShowData3D(d.srcObj3d,d.dstObj3d)
        transformation = d.MatchP2P()
        d.ShowData3D(d.srcObj3d,d.dstObj3d, transformation)
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
    singletest.addTest(TestMatching3D("test_Downsample"))
#    singletest.addTest(TestMatching3D("test_MatchP2P"))
  
    
    unittest.TextTestRunner().run(singletest)