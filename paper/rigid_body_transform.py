import numpy as np
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation as R

def vect_from_skew_symmetric(S):
    """Get the vector from a skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

# Define a 3D rotation for R1 and R2 using scipy's Rotation module for a more general 3D case
R1_3d = R.from_euler('xyz', [45, 30, 60], degrees=True).as_matrix()  # A 3D rotation
R2_3d = R.from_euler('xyz', [40, 35, 55], degrees=True).as_matrix()  # Another 3D rotation close to R1_3d

# Compute the "tangent velocity" as the Lie algebra element relating these two matrices in 3D
R21_3d = R2_3d.T @ R1_3d  # Relative rotation from R2_3d to R1_3d
w_b_skew_3d = logm(R21_3d)  # Use matrix logarithm to find the skew-symmetric matrix
w_b = vect_from_skew_symmetric(w_b_skew_3d)  # Convert the skew-symmetric matrix to a vector

# Define a reflection matrix Ra (Reflections are not in SO(3), but let's simulate an effect with a scaling matrix)
R_random = R.from_euler('xyz', [0, 10, 0], degrees=True).as_matrix()   # Random rotation matrix
Ra_reflect = R_random @ np.diag([1, -1, 1])  # This simulates a reflection by inverting one axis

# Transform both R1_3d and R2_3d using Ra_reflect
R1_transformed = Ra_reflect @ R1_3d @ Ra_reflect
R2_transformed = Ra_reflect @ R2_3d @ Ra_reflect

# Compute the new tangent velocity in the transformed basis
R21_transformed = R2_transformed.T @ R1_transformed

w_b_skew_transformed = logm(R21_transformed)  # New tangent velocity in the transformed basis
w_b_transformed = vect_from_skew_symmetric(w_b_skew_transformed)  # Convert the skew-symmetric matrix to a vector


w_b_reflected = np.linalg.det(Ra_reflect) * Ra_reflect @ w_b

w_b_ref_lie = Ra_reflect @ w_b_skew_3d
w_b_ref = vect_from_skew_symmetric(w_b_ref_lie)

# Apply the conjugate action to the Lie Algebra element for 3D rotations
w_b_skew_3d_conjugated = Ra_reflect @ w_b_skew_3d @ Ra_reflect
w_b_conj = vect_from_skew_symmetric(w_b_skew_3d_conjugated)

print(f"Original tangent velocity: {w_b}"
      f"\nTransformed tangent velocity: {w_b_transformed}"
      f"\n(True value) Reflected |R|R w_B tangent velocity: {w_b_reflected}"
      f"\nReflected R w_B tangent velocity: {w_b_ref}"
      f"\nConjugated tangent velocity: {w_b_conj}")
# Print matrix forms
print(f"Original tangent velocity (skew-symmetric matrix):\n{w_b_skew_3d}"
      f"\nTransformed tangent velocity (skew-symmetric matrix):\n{w_b_skew_transformed}"
      f"\nReflected tangent velocity (skew-symmetric matrix):\n{Ra_reflect @ w_b_skew_3d @ Ra_reflect}")

