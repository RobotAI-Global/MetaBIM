import numpy as np
#from scipy.linalg import lstsq

# --- 1. Example Data (Replace with your actual data) ---
# Create some noisy data based on a known quadratic surface
N = 30
np.random.seed(42)
X = (np.random.rand(N) - 0.5) * 10
Y = (np.random.rand(N) - 0.5) * 10
#Z_true = 0.5*X**2 - 0.2*Y**2 + 0.8*X*Y + 1.5*X - 2.0*Y + 5.0
# parabola with noise
Z_true = 0.5*X**2 + 5.5*Y**2 #- 0.8*X*Y 
Z = Z_true + np.random.normal(0, 0.3, N) # Add some noise

# Prepare the data as column vectors
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# --- 2. Construct the Design Matrix (A) ---
# The design matrix A represents the independent terms of the quadratic equation
# A has 6 columns corresponding to: [x^2, y^2, xy, x, y, 1]
# Each row corresponds to a data point (x_i, y_i)

A = np.c_[X**2, Y**2, X*Y, X, Y, np.ones(N)] 

# A is now of shape (N, 6)
# Z is the dependent variable (the 'b' vector in A*c = Z)

# C holds the coefficients [a, b, c, d, e, f]
C, residual, rank, singular_values = np.linalg.lstsq(A, Z, rcond=None)

# Extract the coefficients
a, b, c, d, e, f = C

print(f"--- Fitted Quadratic Equation ---")
print(f"z = {a:.4f}x^2 + {b:.4f}y^2 + {c:.4f}xy + {d:.4f}x + {e:.4f}y + {f:.4f}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid for the fitted surface
grid_x, grid_y = np.meshgrid(np.linspace(X.min(), X.max(), 30), 
                             np.linspace(Y.min(), Y.max(), 30))

# Evaluate the fitted surface on the grid
grid_z = (a * grid_x**2 + b * grid_y**2 + c * grid_x * grid_y + 
          d * grid_x + e * grid_y + f)

# Plot the result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, color='r', marker='o', label='Original Data Points')
ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, cmap='viridis', label='Fitted Quadratic Surface')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quadratic Surface Fit to 3D Points')
plt.show()
# -----------------------------------

import open3d as o3d

# Assuming you have the fitted coefficients (a, b, c, d, e, f) and a meshgrid (grid_x, grid_y, grid_z) from the NumPy solution.

# --- 1. Original Point Cloud Visualization ---
points = np.vstack((X, Y, Z)).T
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# --- 2. Fitted Surface Mesh Creation ---
# Reshape the grid for mesh creation
vertices = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T

# Create triangles (simplified for a 2D grid surface)
rows, cols = grid_x.shape
triangles = []
for i in range(rows - 1):
    for j in range(cols - 1):
        # Indices in the flattened array
        v1 = i * cols + j
        v2 = i * cols + j + 1
        v3 = (i + 1) * cols + j
        v4 = (i + 1) * cols + j + 1
        
        # Two triangles per quad
        triangles.append([v1, v2, v3])
        triangles.append([v2, v4, v3])

mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(np.asarray(triangles))
)

# Compute normals for shading
mesh.compute_vertex_normals()

# --- 3. Combined Visualization ---
o3d.visualization.draw_geometries([pcd, mesh])

import numpy as np
from numpy.linalg import eig

def find_principal_curvature_and_directions(a, b, c, d, e, f, x0, y0):
    """
    Calculates principal curvatures (kappa1, kappa2) and directions (p1, p2)
    for the quadratic surface z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f 
    at a specific point (x0, y0).
    """

    # --- 1. Calculate Partial Derivatives at (x0, y0) ---
    fx  = 2 * a * x0 + c * y0 + d
    fy  = 2 * b * y0 + c * x0 + e
    fxx = 2 * a
    fyy = 2 * b
    fxy = c
    
    # --- 2. Calculate Fundamental Form Matrices ---
    
    # First Fundamental Form (I)
    I = np.array([
        [1 + fx**2,   fx * fy],
        [fx * fy,     1 + fy**2]
    ])
    
    # Normalization factor for Second Fundamental Form
    W = np.sqrt(1 + fx**2 + fy**2)
    
    # Second Fundamental Form (II)
    II = (1 / W) * np.array([
        [fxx, fxy],
        [fxy, fyy]
    ])
    
    # --- 3. Calculate Weingarten Map (Curvature Tensor) ---
    # W_map = I_inv @ II
    try:
        I_inv = np.linalg.inv(I)
    except np.linalg.LinAlgError:
        print("Error: First Fundamental Form Matrix is singular.")
        return None, None
        
    W_map = I_inv @ II
    
    # --- 4. Find Principal Curvatures and Directions (Eigenvalues and Eigenvectors) ---
    # Eigenvalues of W_map are the principal curvatures (kappa1, kappa2)
    # Eigenvectors of W_map are the principal directions projected onto the (du, dv) plane
    
    eigenvalues, eigenvectors = eig(W_map)
    kappa1, kappa2 = eigenvalues
    
    # The eigenvectors are in the tangent space parameterization (du, dv).
    # The actual directions (in 3D space) are p = (du, dv, fx*du + fy*dv)
    
    # Eigenvector 1 (corresponding to kappa1)
    du1, dv1 = eigenvectors[:, 0]
    p1 = np.array([du1, dv1, fx * du1 + fy * dv1])
    
    # Eigenvector 2 (corresponding to kappa2)
    du2, dv2 = eigenvectors[:, 1]
    p2 = np.array([du2, dv2, fx * du2 + fy * dv2])
    
    # Normalize the direction vectors
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    return (kappa1, kappa2), (p1, p2)


# --- EXAMPLE USAGE ---

# Coefficients from the previous fit result (example values)
# a_fit = 0.5091
# b_fit = -0.0891
# c_fit = 0.8012
# d_fit = 1.2487
# e_fit = -2.7698
# f_fit = 6.5546

a_fit, b_fit, c_fit, d_fit, e_fit, f_fit = C 

# Evaluate at a point, e.g., the origin (0, 0)
x_point, y_point = 0.0, 0.0

curvatures, directions = find_principal_curvature_and_directions(
    a_fit, b_fit, c_fit, d_fit, e_fit, f_fit, x_point, y_point
)

if curvatures:
    kappa1, kappa2 = curvatures
    p1, p2 = directions
    
    print(f"\n--- Results at Point ({x_point}, {y_point}) ---")
    print(f"Mean Curvature (H): {(kappa1 + kappa2) / 2:.4f}")
    print(f"Gaussian Curvature (K): {kappa1 * kappa2:.4f}")
    print("\nPrincipal Curvatures:")
    print(f"Maximal Curvature (κ1): {kappa1:.6f}")
    print(f"Minimal Curvature (κ2): {kappa2:.6f}")
    
    print("\nPrincipal Directions (3D Vector):")
    print(f"Direction 1 (Max Curv.): {p1}")
    print(f"Direction 2 (Min Curv.): {p2}")
    
    # Check for orthogonality (p1 . p2 should be near zero)
    ortho_check = np.dot(p1, p2)
    print(f"\nOrthogonality Check (p1 · p2): {ortho_check:.2e} (should be near 0)")