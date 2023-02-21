
# # Show how to do matching usinh open cv PFF

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

import cv2 as cv
import numpy as np

N = 2
modelname = "parasaurolophus_6700"
scenename = "rs1_normals"

detector = cv.ppf_match_3d_PPF3DDetector(0.025, 0.05)

print('Loading model...')
pc = cv.ppf_match_3d.loadPLYSimple("%s.ply" % modelname, 1)


print('Training...')
detector.trainModel(pc)

print('Loading scene...')
pcTest = cv.ppf_match_3d.loadPLYSimple("%s.ply" % scenename, 1)

print('Matching...')
#results = detector.match(pcTest, 1.0/40.0, 0.05)
results = detector.match(pcTest, 0.5, 0.05)

print('Performing ICP...')
icp = cv.ppf_match_3d_ICP(100)
_, results = icp.registerModelToScene(pc, pcTest, results[:N])

print("Poses: ")
resultFileName = "%sPCTrans.ply" % modelname
for i, result in enumerate(results):
    #result.printPose()
    print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
    if i == 0:
        pct = cv.ppf_match_3d.transformPCPose(pc, result.pose)
        cv.ppf_match_3d.writePLY(pct, resultFileName)

#%% using Open3d
import open3d as o3d
import numpy as np

modelname = "parasaurolophus_6700"
scenename = "rs1_normals"
resultFileName = "%sPCTrans.ply" % modelname
pcd_read = o3d.io.read_point_cloud('%s.ply' %modelname)
o3d.visualization.draw_geometries([pcd_read])
pcd_read = o3d.io.read_point_cloud('%s.ply' %scenename)
o3d.visualization.draw_geometries([pcd_read])
pcd_read = o3d.io.read_point_cloud(resultFileName)
o3d.visualization.draw_geometries([pcd_read])


#%% show not working
import cv2 as cv
import numpy as np
modelname = "parasaurolophus_6700"
resultFileName = "%sPCTrans.ply" % modelname
def load_bunny():
    with open(cv.samples.findFile(resultFileName), 'r') as f:
        s = f.read()
    ligne = s.split('\n')
    if len(ligne) == 5753:
        pts3d = np.zeros(shape=(1,1889,3), dtype=np.float32)
        pts3d_c = 255 * np.ones(shape=(1,1889,3), dtype=np.uint8)
        pts3d_n = np.ones(shape=(1,1889,3), dtype=np.float32)
        for idx in range(12,1889):
            d = ligne[idx].split(' ')
            pts3d[0,idx-12,:] = (float(d[0]), float(d[1]), float(d[2]))
    pts3d = 5 * pts3d
    return cv.viz_WCloud(pts3d)

myWindow = cv.viz_Viz3d("Coordinate Frame")
axe = cv.viz_WCoordinateSystem()
myWindow.showWidget("axe",axe)

cam_pos =  (3.0, 3.0, 3.0)
cam_focal_point = (3.0,3.0,2.0)
cam_y_dir = (-1.0,0.0,0.0)
cam_pose = cv.viz.makeCameraPose(cam_pos, cam_focal_point, cam_y_dir)
print("OK")
transform = cv.viz.makeTransformToGlobal((0.0,-1.0,0.0), (-1.0,0.0,0.0), (0.0,0.0,-1.0), cam_pos)
pw_bunny = load_bunny()
cloud_pose = cv.viz_Affine3d()
cloud_pose = cloud_pose.translate((0, 0, 3))
cloud_pose_global = transform.product(cloud_pose)

cpw = cv.viz_WCameraPosition(0.5)
cpw_frustum = cv.viz_WCameraPosition(0.3)
myWindow.showWidget("CPW", cpw);
myWindow.showWidget("CPW_FRUSTUM", cpw_frustum)
myWindow.setViewerPose(cam_pose)
myWindow.showWidget("bunny", pw_bunny, cloud_pose_global)
#myWindow.setWidgetPosePy("bunny")
myWindow.spin();
print("Last event loop is over")
# %%
