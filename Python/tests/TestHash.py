# Test Open3D Hash Function

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

import open3d as o3d
import open3d.core as o3c
#import cv2 as cv
import numpy as np
import time

def get_box_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    #mesh1 = o3d.geometry.TriangleMesh.create_box()
    mesh.transform(T)
    # debug - show mesh
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    #o3d.visualization.draw([mesh])
    #print(mesh)
    return mesh

def get_vertex(vtype = 1):
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

    elif vtype == 11: # random with 1 match
        point_data = np.random.rand(32,3)*100

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
    pcd.paint_uniform_color([0.3, 0.3, 0.3])
    #o3d.visualization.draw([pcd])
    return pcd

def closest_point(pcd):
    # find closest points
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print("Paint the 0 point red.")
    pcd.colors[0] = [1, 0, 0]

    print("Find its 3 nearest neighbors, and paint them blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], 4)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    print(idx)

    print("Paint the 7 point red.")
    pcd.colors[7] = [1, 0, 0]
    
    print("Find its neighbors with distance less than 2, and paint them green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[7], 2.5)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    print(idx)

    print("Visualize the point cloud.")
    """ o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0 9768, 0.2024])"""
    
    o3d.visualization.draw([pcd])
    
def do_distance_hash(points):
    capacity = 10
    device = o3c.Device('cpu:0')
    
    # compute distances
    vals = []
    keys = []
    for i in range(len(points)):
        for j in range(len(points)):
        
            dist_ij = np.linalg.norm(points[i] - points[j])
            keys.append(dist_ij)
            vals.append((i,j))    
    
    # define hash
    hashmap = o3c.HashMap(capacity,
                      key_dtype=o3c.int64,
                      key_element_shape=(1,),
                      value_dtype=o3c.int64,
                      value_element_shape=(1,),
                      device=device)  
    
    keys = o3c.Tensor(keys)
    vals = o3c.Tensor(vals)
    buf_indices, masks = hashmap.insert(keys, vals)
    #print(buf_indices)

    print('masks: \n', masks)
    print('inserted keys: \n', keys[masks])  
    
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
      
def do_distance_hash_python(points):
    # compute distances
    dist_dict = {}
    for i in range(len(points)):
        for j in range(len(points)):
        
            dist_ij = np.linalg.norm(points[i] - points[j])
            dkey = dist2key(dist_ij) #np.round(dist_ij*10)/10
            add_value(dist_dict, dkey, (i,j))
            #keys.append(dist_ij)
            #vals.append((i,j)) 
    return dist_dict

def get_match_pairs(dist_hash, points, i, j):
    # extract matching pairs from hash dictionary
    dist_points = np.linalg.norm(points[i] - points[j])
    dkey  = dist2key(dist_points) #np.round(dist_points*10)/10
    pairs = dist_hash.get(dkey)
    return pairs

def pairs_intersect(pairs_ij, pairs_ik):
    # intersect common indices of the arrays
    # compare the first column and report when matching
    pairs_ij = np.array(pairs_ij)
    pairs_ik = np.array(pairs_ik)
    pairs_jk = []
    for k in range(pairs_ij.shape[0]):
        ind = np.where(pairs_ik[:,0] == pairs_ij[k,0])[0]
        if ind is None:
            continue
        temp_jk = [pairs_ij[k,1]*np.ones((len(ind),)).reshape((-1,1)), pairs_ik[ind,1].reshape((-1,1))]
        pairs_jk.append(temp_jk)
        
    return np.array(pairs_jk).squeeze()

def sets_intersect(pairs_ij, pairs_ik):
    # intersect common indices of the sets
    # make all the combinations when the first column matches
    
    pairs_jk = set()
    
    for sij in pairs_ij:
        for sik in pairs_ik:
            if sij[0] == sik[0]:
                pairs_jk.add((sij[0], sij[1] ,sik[1]))

    return pairs_jk

def dict_intersect(pairs_ij, pairs_ik):
    # intersect common indices of the sets
    # make all the combinations when the first column matches
    
    pairs_jk = {}
    
    for sij in pairs_ij:
        for sik in pairs_ik:
            if sij[0] == sik[0]:
                add_value(pairs_jk,(sij[1] ,sik[1]),sij[0])

    return pairs_jk

def match_triple_dict(pairs_ij, pairs_ik, pairs_jk):
    # check matching
    pairs_jk_est  = dict_intersect(pairs_ij, pairs_ik)

    #print(pairs_jk_est.keys())
    pairs_common = pairs_jk_est.keys() & pairs_jk
    print(pairs_common)
    triples = set()
    for k in pairs_common:
        if isinstance(pairs_jk_est[k],list):
            for v in pairs_jk_est[k]:
                triples.add((v,k[0],k[1]))
        else:
            triples.add((pairs_jk_est[k],k[0],k[1]))
    
            
    return triples

def get_match_triples(dist_hash, points, i, j, k):
    # extract matching triples from hash dictionary
    dkey_ij  = dist2key(np.linalg.norm(points[i] - points[j]))
    dkey_ik  = dist2key(np.linalg.norm(points[i] - points[k]))
    dkey_jk  = dist2key(np.linalg.norm(points[j] - points[k]))
    pairs_ij = dist_hash.get(dkey_ij)
    pairs_ik = dist_hash.get(dkey_ik)
    pairs_jk = dist_hash.get(dkey_jk)
    # intersection
    #pairs_ijk = sets_intersect(pairs_ij, pairs_ik)
    #pairs_jki = sets_intersect(pairs_ik, pairs_jk)
    #inter_jk = np.intersect1d(pairs_ij[:,1], pairs_ik[:,1])
    cycles_ijk = match_triple_dict(pairs_ij, pairs_ik, pairs_jk)
    
    print("set ij:",pairs_ij)
    print("set ik:",pairs_ik)
    print("set jk:",pairs_jk)
    print('cycle ijk: ',cycles_ijk)  
    #print(cycles_ijk)
    return cycles_ijk

def find_max_points_in_hash(dist_hash):
    # runs on all the key pairs in the hash and finds the max number 
    mv      = 0
    for k in dist_hash:
        vlist = dist_hash[k]
        for e in vlist:
            mv = np.maximum(mv, e[0])
    return mv

def point_select(dist_hash, pnum = 1):
    # select the lowest probability tuples from the hash
    MAX_POINT_NUM   = find_max_points_in_hash(dist_hash)
    print('MAX_POINT_NUM %d' %MAX_POINT_NUM)
    count_per_node  = np.zeros(MAX_POINT_NUM+1)
    for k in dist_hash:
        vlist = dist_hash[k]
        for e in vlist:
            #print(e)
            n1 = e[0]
            count_per_node[n1] += len(vlist)
            n2 = e[1]
            count_per_node[n2] += len(vlist)
         
    #print(dist_hash)   
    print(count_per_node)
    #This returns the k-smallest values. Note that these may not be in sorted order.
    MIN_NUMBER = 3
    idx = np.argpartition(count_per_node, MIN_NUMBER)
    print(idx)
    print(count_per_node[idx[:MIN_NUMBER]])
        
    return count_per_node

                       
# =========================================    
def test_closest_point():
    # testing the closest point
    pcd = get_pcd_from_vertex()
    closest_point(pcd)

def test_dist_hash():
    # testing the hash 
    pcd = get_pcd_from_vertex()
    do_distance_hash(pcd.points)
    
def test_python_hash():
    # Dictionary of names and phone numbers
    phone_details = {   'Mathew': 212323,
                    'Ritika': 334455,
                    'John'  : 345323 }
    # Append a value to the existing key 
    add_value(phone_details, 'John', 111223)
    # Append a value to the existing key
    add_value(phone_details, 'John', 333444)
    for key, value in phone_details.items():
        print(key, ' - ', value) 

def test_dist_hash_python():
    # testing the hash 
    pcd = get_pcd_from_vertex()
    dct = do_distance_hash_python(pcd.points)        
    print(dct)
    
def test_get_pairs():
    # testing the hash 
    pcd = get_pcd_from_vertex()
    dct = do_distance_hash_python(pcd.points)   
    ii, jj = 0,7   
    pairs = get_match_pairs(dct, pcd.points, ii, jj)  
    print(pairs)
    
def test_pairs_intersect():
    # test different combinations of indices
    pij = [[0,1],[0,2],[1,3],[1,4]]
    pik = [[0,4],[1,5],[1,7],[2,6]]
    pjk = pairs_intersect(pij,pik)
    print("set ij:",pij)
    print("set ik:",pik)
    print('intersect jk: ',pjk)
       
def test_sets_intersect():
    # test different combinations of indices
    pij = set([(0,1),(0,2),(1,3),(1,4)])
    pik = set([(0,4),(1,5),(1,7),(2,6),(3,4)])
    pjk = sets_intersect(pij,pik)
    
    print("set ij:",pij)
    print("set ik:",pik)
    print('intersect ijk: ',pjk) 
    
def test_dicts_intersect():
    # test different combinations of indices
    pij = set([(0,1),(0,2),(1,3),(1,4)])
    pik = set([(0,4),(1,5),(1,7),(2,6),(3,4)])
    pjk = dict_intersect(pij,pik).keys()
    print("set ij:",pij)
    print("set ik:",pik)
    print('intersect ijk: ',pjk)  
    
def test_match_dicts():
    # test different combinations of indices
    pij = set([(0,1),(0,2),(1,3),(1,4),(2,3)])
    pik = set([(0,4),(1,5),(1,7),(2,5),(2,6),(3,4)])
    pjk = set([(1,4),(3,5),(2,5)])
    pijk = match_triple_dict(pij,pik,pjk)
    print("set ij:",pij)
    print("set ik:",pik)
    print("set jk:",pjk)
    print('cycle ijk: ',pijk)     

       
def test_match_triples():
    # matching triples
    dat = get_vertex(12)
    pcd = get_pcd_from_vertex(dat)
    dct = do_distance_hash_python(pcd.points)   
    i, j, k = 0,2,1  
    triples = get_match_triples(dct, pcd.points, i, j, k)  
    #print(triples)
    
def test_point_select(pnum = 3):
    # testing point selection from the distances 
    dat = get_vertex(11)
    pcd = get_pcd_from_vertex(dat)
    dct = do_distance_hash_python(pcd.points) 
    points = point_select(dct)       

if __name__ == "__main__":

    #get_box_mesh()
    #get_pcd_from_vertex()
    #test_closest_point() # OK
    #test_dist_hash()
    #test_python_hash()
    #test_dist_hash_python() # ok
    #test_get_pairs()
    #test_pairs_intersect() #nok
    #test_sets_intersect()
    #test_dicts_intersect()
    #test_match_dicts()  # ok
    #test_match_triples() # OK
    test_point_select()

    