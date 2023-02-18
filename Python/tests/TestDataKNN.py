

# # Show hopw to do KNN

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
from sklearn.neighbors import KDTree
import numpy as np

#%% Query for neighbors within a given radius


rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)     
print(tree.query_radius(X[:1], r=0.3, count_only=True))

ind = tree.query_radius(X[:1], r=0.3)  
print(ind)  # indices of neighbors within distance 0.3
