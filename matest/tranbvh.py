import os
import phygamenn.utils.BVH as BVH
import phygamenn.utils.Animation as Animation
# from phygamenn.utils.Quaternions import Quaternions
from phygamenn.utils.InverseKinematics import JacobianInverseKinematics
import pkg_resources
import numpy as np
# import math
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
import json


def retarget(deepm_temp_file, deepm_bvh_file, bvh_file):
    print('Processing %s' % bvh_file)

    src_anim, src_names, src_frames = BVH.load(bvh_file)
    src_gps = Animation.positions_global(src_anim)

    tar_rest, tar_names, _ = BVH.load(deepm_temp_file)

    anim = tar_rest.copy()
    anim.positions = anim.positions.repeat(len(src_gps), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(src_gps), axis=0)

    anim.positions[:, 0] = src_anim.positions[:, 0]
    anim.rotations[:, 0] = src_anim.rotations[:, 0]

    mapping = {
        13: 2, 14: 3, 15: 4, 16: 5,  # left leg
        9: 7, 10: 8, 11: 9, 12: 10,  # right leg
        6: 18, 7: 19, 8: 20,  # left arm
        3: 25, 4: 26, 5: 27,  # right arm
        1: 12, 2: 15
    }

    targetmap = {}

    for k in mapping:
        targetmap[k] = src_gps[:, mapping[k]]

    weight_trans = np.zeros(anim.shape[1])
    weight_trans[0:4] = 1.0

    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=False,
                                   translate=True, weights_translate=weight_trans)
    ik()

    result = tar_rest.copy()
    result.positions = anim.positions
    result.positions[:, 0, :] /= 2.0
    result.offsets /= 2.0
    result.rotations.qs = anim.rotations.qs
    BVH.save(deepm_bvh_file, result, tar_names, src_frames)


def joint_from_3d_to_1d(joint_id, quats):
    # The joint must only have one parent and one child.
    qs = quats.copy()
    parent_id = joint_id - 1
    child_id = joint_id + 1

    joint_yzy = R.from_quat(
        quat=qs[:, joint_id, [1, 2, 3, 0]]).as_euler('yzy', degrees=True)
    parent_dcm = R.from_quat(quat=qs[:, parent_id, [1, 2, 3, 0]]).as_dcm()
    child_dcm = R.from_quat(quat=qs[:, child_id, [1, 2, 3, 0]]).as_dcm()

    H1 = R.from_euler('y', joint_yzy[:, 0], degrees=True).as_dcm()
    H2 = R.from_euler('z', joint_yzy[:, 1], degrees=True).as_dcm()
    H3 = R.from_euler('y', joint_yzy[:, 2], degrees=True).as_dcm()

    parent_qs = R.from_dcm(np.matmul(parent_dcm, H3)
                           ).as_quat()[..., [3, 0, 1, 2]]
    child_qs = R.from_dcm(np.matmul(H1, child_dcm)
                          ).as_quat()[..., [3, 0, 1, 2]]
    joint_qs = R.from_dcm(H2).as_quat()[..., [3, 0, 1, 2]]

    qs[:, joint_id] = joint_qs
    qs[:, parent_id] = parent_qs
    qs[:, child_id] = child_qs

    return qs


def simplify_and_scale(deepm_bvh_file, bvh_file, scale_factor=1.0):
    src_anim, src_names, src_rate = BVH.load(deepm_bvh_file)
    qs = src_anim.rotations.qs.copy()

    qs = joint_from_3d_to_1d(10, qs)
    qs = joint_from_3d_to_1d(14, qs)
    qs = joint_from_3d_to_1d(4, qs)
    qs = joint_from_3d_to_1d(7, qs)

    result = src_anim.copy()
    result.positions[:, 0, :] = result.positions[:, 0, :] * scale_factor
    result.positions[:, 1:, :] = np.zeros_like(result.positions[:, 1:, :])
    result.offsets = result.offsets * scale_factor
    result.rotations.qs = qs
    BVH.save(bvh_file, result, src_names, src_rate)


def bvh2json(bvh_file, json_file, scale_factor=0.1):
    src_anim, src_names, src_rate = BVH.load(bvh_file)
    qs = src_anim.rotations.qs
    positions = src_anim.positions

    frames = np.zeros((qs.shape[0], 44))
    frames[:, 0:1] = 1 / 60.0
    frames[:, 1:4] = positions[:, 0, :] * scale_factor  # hip position xyz 3D
    frames[:, 4:8] = qs[:, 0, :]  # hip rotation4D
    frames[:, 8:12] = qs[:, 1, :]  # chest rotation4D
    frames[:, 12:16] = qs[:, 2, :]  # neck rotation4D
    frames[:, 16:20] = qs[:, 9, :]  # right hip rotation4D
    frames[:, 20:21] = R.from_quat(qs[:, 10, [1, 2, 3, 0]]).as_euler("ZYX")[
        :, 0:1]  # right knee rotation1D
    frames[:, 21:25] = qs[:, 11, :]  # right ankle rotation4D
    frames[:, 25:29] = qs[:, 3, :]  # right shoulder rotation4D
    frames[:, 29:30] = R.from_quat(qs[:, 4, [1, 2, 3, 0]]).as_euler("ZYX")[
        :, 0:1]  # right elbow rotation1D
    frames[:, 30:34] = qs[:, 13, :]  # left hip rotation4D
    frames[:, 34:35] = R.from_quat(qs[:, 14, [1, 2, 3, 0]]).as_euler("ZYX")[
        :, 0:1]  # left knee rotation1D
    frames[:, 35:39] = qs[:, 15, :]  # left ankle rotation4D
    frames[:, 39:43] = qs[:, 6, :]  # left shoulder rotation4D
    frames[:, 43:44] = R.from_quat(qs[:, 7, [1, 2, 3, 0]]).as_euler("ZYX")[
        :, 0:1]  # left elbow rotation1D

    frames = frames[::2].tolist()

    output = {}
    output["Loop"] = "none"
    output["Frames"] = frames
    with open(json_file, 'w') as f:
        json.dump(output, f)
    return output


if __name__ == '__main__':
    #pnum = 2
    bvh2json("TestAA.bvh", "TestAA.json");
    print("---BVH to JSON---")
    #po.close()
    #po.join()
