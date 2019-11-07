import numpy as np
from Quaternions import Quaternions
from scipy.spatial.transform import Rotation as R

rotations = np.array([ 170.537549 , -1.051628,-175.45039])
print(rotations)
quantt = Quaternions.from_euler(
        np.radians(rotations), order='zyx', world=False)
print(quantt.qs)