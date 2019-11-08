import numpy as np
import copy as copy
import tensorflow as tf
from enum import Enum
import os
import time
import gym
from gym.spaces import Box
from PhysicsCore import DeepMimicCore

from phygamenn.utils.MPISolver import MPISolver
from phygamenn.utils.Logger import Logger
import phygamenn.utils.MPIUtils as MPIUtil
import inspect as inspect
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

INVALID_IDX = -1


class Terminate(Enum):
    Null = 0
    Fail = 1
    Succ = 2


class ActionSpace(Enum):
    Null = 0
    Continuous = 1
    Discrete = 2


class Path(object):
    def __init__(self):
        self.clear()
        return

    def pathlength(self):
        return len(self.actions)

    def is_valid(self):
        valid = True
        length = self.pathlength()
        # 状态是多保存了一个的
        valid &= len(self.states) == length + 1
        valid &= len(self.goals) == length + 1
        valid &= len(self.actions) == length
        valid &= len(self.logps) == length
        valid &= len(self.rewards) == length
        valid &= len(self.flags) == length

        # print(len(self.states), len(self.goals), len(self.actions), len(self.logps), len(self.rewards), len(self.flags))

        return valid

    def check_vals(self):
        for vals in [self.states, self.goals, self.actions, self.logps, self.rewards]:
            for v in vals:
                if not np.isfinite(v).all():
                    return False
        return True

    def clear(self):
        self.states = []
        self.goals = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.flags = []
        self.terminate = Terminate.Null
        return

    def get_pathlen(self):
        return len(self.rewards)

    def calc_return(self):
        return sum(self.rewards)


class Normalizer(object):
    CHECK_SYNC_COUNT = 50000  # check synchronization after a certain number of entries

    # these group IDs must be the same as those in CharController.h
    NORM_GROUP_SINGLE = 0
    NORM_GROUP_NONE = -1

    class Group(object):
        def __init__(self, id, indices):
            self.id = id
            self.indices = indices
            return

    def __init__(self, size, groups_ids=None, eps=0.02, clip=np.inf):
        self.eps = eps
        self.clip = clip
        self.mean = np.zeros(size)
        self.mean_sq = np.zeros(size)
        self.std = np.ones(size)
        self.count = 0
        self.groups = self._build_groups(groups_ids)

        self.new_count = 0
        self.new_sum = np.zeros_like(self.mean)
        self.new_sum_sq = np.zeros_like(self.mean_sq)
        return

    def record(self, x):
        size = self.get_size()
        is_array = isinstance(x, np.ndarray)
        if not is_array:
            assert(size == 1)
            x = np.array([[x]])

        assert x.shape[-1] == size, \
            Logger.print('Normalizer shape mismatch, expecting size {:d}, but got {:d}'.format(
                size, x.shape[-1]))
        x = np.reshape(x, [-1, size])

        self.new_count += x.shape[0]
        self.new_sum += np.sum(x, axis=0)
        self.new_sum_sq += np.sum(np.square(x), axis=0)
        return

    def update(self):
        new_count = MPIUtil.reduce_sum(self.new_count)
        new_sum = MPIUtil.reduce_sum(self.new_sum)
        new_sum_sq = MPIUtil.reduce_sum(self.new_sum_sq)

        new_total = self.count + new_count
        if (self.count // self.CHECK_SYNC_COUNT != new_total // self.CHECK_SYNC_COUNT):
            assert self.check_synced(), Logger.print(
                'Normalizer parameters desynchronized')

        if new_count > 0:
            new_mean = self._process_group_data(new_sum / new_count, self.mean)
            new_mean_sq = self._process_group_data(
                new_sum_sq / new_count, self.mean_sq)
            w_old = float(self.count) / new_total
            w_new = float(new_count) / new_total

            self.mean = w_old * self.mean + w_new * new_mean
            self.mean_sq = w_old * self.mean_sq + w_new * new_mean_sq
            self.count = new_total
            self.std = self.calc_std(self.mean, self.mean_sq)

            self.new_count = 0
            self.new_sum.fill(0)
            self.new_sum_sq.fill(0)

        return

    def get_size(self):
        return self.mean.size

    def set_mean_std(self, mean, std):
        size = self.get_size()
        is_array = isinstance(mean, np.ndarray) and isinstance(std, np.ndarray)

        if not is_array:
            assert(size == 1)
            mean = np.array([mean])
            std = np.array([std])

        assert len(mean) == size and len(std) == size, \
            Logger.print('Normalizer shape mismatch, expecting size {:d}, but got {:d} and {:d}'.format(
                size, len(mean), len(std)))

        self.mean = mean
        self.std = std
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def normalize(self, x):
        norm_x = (x - self.mean) / self.std
        norm_x = np.clip(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize(self, norm_x):
        x = norm_x * self.std + self.mean
        return x

    def calc_std(self, mean, mean_sq):
        var = mean_sq - np.square(mean)
        # some time floating point errors can lead to small negative numbers
        var = np.maximum(var, 0)
        std = np.sqrt(var)
        std = np.maximum(std, self.eps)
        return std

    def calc_mean_sq(self, mean, std):
        return np.square(std) + np.square(self.mean)

    def check_synced(self):
        synced = True
        if MPIUtil.is_root_proc():
            vars = np.concatenate([self.mean, self.mean_sq])
            MPIUtil.bcast(vars)
        else:
            vars_local = np.concatenate([self.mean, self.mean_sq])
            vars_root = np.empty_like(vars_local)
            MPIUtil.bcast(vars_root)
            synced = (vars_local == vars_root).all()
        return synced

    def _build_groups(self, groups_ids):
        groups = []
        if groups_ids is None:
            curr_id = self.NORM_GROUP_SINGLE
            curr_list = np.arange(self.get_size()).astype(np.int32)
            groups.append(self.Group(curr_id, curr_list))
        else:
            ids = np.unique(groups_ids)
            for id in ids:
                curr_list = np.nonzero(groups_ids == id)[0].astype(np.int32)
                groups.append(self.Group(id, curr_list))

        return groups

    def _process_group_data(self, new_data, old_data):
        proc_data = new_data.copy()
        for group in self.groups:
            if group.id == self.NORM_GROUP_NONE:
                proc_data[group.indices] = old_data[group.indices]
            elif group.id != self.NORM_GROUP_SINGLE:
                avg = np.mean(new_data[group.indices])
                proc_data[group.indices] = avg
        return proc_data


class TFNormalizer(Normalizer):

    def __init__(self, sess, scope, size, groups_ids=None, eps=0.02, clip=np.inf):
        self.sess = sess
        self.scope = scope
        super().__init__(size, groups_ids, eps, clip)

        with tf.variable_scope(self.scope):
            self._build_resource_tf()
        return

    # initialze count when loading saved values so that things don't change to quickly during updates
    def load(self):
        self.count = self.count_tf.eval()[0]
        self.mean = self.mean_tf.eval()
        self.std = self.std_tf.eval()
        self.mean_sq = self.calc_mean_sq(self.mean, self.std)
        return

    def update(self):
        super().update()
        self._update_resource_tf()
        return

    def set_mean_std(self, mean, std):
        super().set_mean_std(mean, std)
        self._update_resource_tf()
        return

    def normalize_tf(self, x):
        norm_x = (x - self.mean_tf) / self.std_tf
        norm_x = tf.clip_by_value(norm_x, -self.clip, self.clip)
        return norm_x

    def unnormalize_tf(self, norm_x):
        x = norm_x * self.std_tf + self.mean_tf
        return x

    def _build_resource_tf(self):
        self.count_tf = tf.get_variable(dtype=tf.int32, name='count', initializer=np.array([
                                        self.count], dtype=np.int32), trainable=False)
        self.mean_tf = tf.get_variable(
            dtype=tf.float32, name='mean', initializer=self.mean.astype(np.float32), trainable=False)
        self.std_tf = tf.get_variable(
            dtype=tf.float32, name='std', initializer=self.std.astype(np.float32), trainable=False)

        self.count_ph = tf.get_variable(
            dtype=tf.int32, name='count_ph', shape=[1])
        self.mean_ph = tf.get_variable(
            dtype=tf.float32, name='mean_ph', shape=self.mean.shape)
        self.std_ph = tf.get_variable(
            dtype=tf.float32, name='std_ph', shape=self.std.shape)

        self._update_op = tf.group(
            self.count_tf.assign(self.count_ph),
            self.mean_tf.assign(self.mean_ph),
            self.std_tf.assign(self.std_ph)
        )
        return

    def _update_resource_tf(self):
        feed = {
            self.count_ph: np.array([self.count], dtype=np.int32),
            self.mean_ph: self.mean,
            self.std_ph: self.std
        }
        self.sess.run(self._update_op, feed_dict=feed)
        return


class ReplayBuffer(object):
    TERMINATE_KEY = 'terminate'
    PATH_START_KEY = 'path_start'
    PATH_END_KEY = 'path_end'

    def __init__(self, buffer_size):
        assert buffer_size > 0

        self.buffer_size = buffer_size
        self.total_count = 0
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0
        self._sample_buffers = dict()
        self.buffers = None

        self.clear()
        return

    def sample(self, n):
        curr_size = self.get_current_size()
        assert curr_size > 0

        idx = np.empty(n, dtype=int)
        # makes sure that the end states are not sampled
        for i in range(n):
            while True:
                curr_idx = np.random.randint(0, curr_size, size=1)[0]
                curr_idx += self.buffer_tail
                curr_idx = np.mod(curr_idx, self.buffer_size)

                if not self.is_path_end(curr_idx):
                    break
            idx[i] = curr_idx

        return idx

    def sample_filtered(self, n, key):
        assert key in self._sample_buffers
        curr_buffer = self._sample_buffers[key]
        idx = curr_buffer.sample(n)
        return idx

    def count_filtered(self, key):
        curr_buffer = self._sample_buffers[key]
        return curr_buffer.count

    def get(self, key, idx):
        return self.buffers[key][idx]

    def get_all(self, key):
        return self.buffers[key]

    def get_idx_filtered(self, key):
        assert key in self._sample_buffers
        curr_buffer = self._sample_buffers[key]
        idx = curr_buffer.slot_to_idx[:curr_buffer.count]
        return idx

    def get_path_start(self, idx):
        return self.buffers[self.PATH_START_KEY][idx]

    def get_path_end(self, idx):
        return self.buffers[self.PATH_END_KEY][idx]

    def get_pathlen(self, idx):
        is_array = isinstance(idx, np.ndarray) or isinstance(idx, list)
        if not is_array:
            idx = [idx]

        n = len(idx)
        start_idx = self.get_path_start(idx)
        end_idx = self.get_path_end(idx)
        pathlen = np.empty(n, dtype=int)

        for i in range(n):
            curr_start = start_idx[i]
            curr_end = end_idx[i]
            if curr_start < curr_end:
                curr_len = curr_end - curr_start
            else:
                curr_len = self.buffer_size - curr_start + curr_end
            pathlen[i] = curr_len

        if not is_array:
            pathlen = pathlen[0]

        return pathlen

    def is_valid_path(self, idx):
        start_idx = self.get_path_start(idx)
        valid = start_idx != INVALID_IDX
        return valid

    def store(self, path):
        start_idx = INVALID_IDX
        n = path.pathlength()
        if (n > 0):
            assert path.is_valid()

            if path.check_vals():
                if self.buffers is None:
                    self._init_buffers(path)

                idx = self._request_idx(n + 1)
                self._store_path(path, idx)
                self._add_sample_buffers(idx)

                self.num_paths += 1
                self.total_count += n + 1
                start_idx = idx[0]
            else:
                Logger.print('Invalid path data value detected')

        return start_idx

    def clear(self):
        self.buffer_head = 0
        self.buffer_tail = INVALID_IDX
        self.num_paths = 0

        for key in self._sample_buffers:
            self._sample_buffers[key].clear()
        return

    def get_next_idx(self, idx):
        next_idx = np.mod(idx + 1, self.buffer_size)
        return next_idx

    def is_terminal_state(self, idx):
        terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
        # 也用到了env
        terminate = terminate_flags != Terminate.Null.value
        is_end = self.is_path_end(idx)
        terminal_state = np.logical_and(terminate, is_end)
        return terminal_state

    def check_terminal_flag(self, idx, flag):
        terminate_flags = self.buffers[self.TERMINATE_KEY][idx]
        terminate = terminate_flags == flag.value
        return terminate

    def is_path_end(self, idx):
        is_end = self.buffers[self.PATH_END_KEY][idx] == idx
        return is_end

    def add_filter_key(self, key):
        assert self.get_current_size() == 0
        if key not in self._sample_buffers:
            self._sample_buffers[key] = SampleBuffer(self.buffer_size)
        return

    def get_current_size(self):
        if self.buffer_tail == INVALID_IDX:
            return 0
        elif self.buffer_tail < self.buffer_head:
            return self.buffer_head - self.buffer_tail
        else:
            return self.buffer_size - self.buffer_tail + self.buffer_head

    def _check_flags(self, key, flags):
        return (flags & key) == key

    def _add_sample_buffers(self, idx):
        flags = self.buffers['flags']
        for key in self._sample_buffers:
            curr_buffer = self._sample_buffers[key]
            filter_idx = [i for i in idx if (self._check_flags(key, flags[i]) and not self.is_path_end(i))]
            curr_buffer.add(filter_idx)
        return

    def _free_sample_buffers(self, idx):
        for key in self._sample_buffers:
            curr_buffer = self._sample_buffers[key]
            curr_buffer.free(idx)
        return

    def _init_buffers(self, path):
        self.buffers = dict()
        self.buffers[self.PATH_START_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int)
        self.buffers[self.PATH_END_KEY] = INVALID_IDX * np.ones(self.buffer_size, dtype=int)

        for key in dir(path):
            val = getattr(path, key)
            if not key.startswith('__') and not inspect.ismethod(val):
                if key == self.TERMINATE_KEY:
                    self.buffers[self.TERMINATE_KEY] = np.zeros(shape=[self.buffer_size], dtype=int)
                else:
                    val_type = type(val[0])
                    is_array = val_type == np.ndarray
                    if is_array:
                        shape = [self.buffer_size, val[0].shape[0]]
                        dtype = val[0].dtype
                    else:
                        shape = [self.buffer_size]
                        dtype = val_type
                    self.buffers[key] = np.zeros(shape, dtype=dtype)
        return

    def _request_idx(self, n):
        assert n + 1 < self.buffer_size  # bad things can happen if path is too long

        remainder = n
        idx = []

        start_idx = self.buffer_head
        while remainder > 0:
            end_idx = np.minimum(start_idx + remainder, self.buffer_size)
            remainder -= (end_idx - start_idx)

            free_idx = list(range(start_idx, end_idx))
            self._free_idx(free_idx)
            idx += free_idx
            start_idx = 0

        self.buffer_head = (self.buffer_head + n) % self.buffer_size
        return idx

    def _free_idx(self, idx):
        assert(idx[0] <= idx[-1])
        n = len(idx)
        if self.buffer_tail != INVALID_IDX:
            update_tail = idx[0] <= idx[-1] and idx[0] <= self.buffer_tail and idx[-1] >= self.buffer_tail
            update_tail |= idx[0] > idx[-1] and (idx[0] <= self.buffer_tail or idx[-1] >= self.buffer_tail)
            if update_tail:
                i = 0
                while i < n:
                    curr_idx = idx[i]
                    if self.is_valid_path(curr_idx):
                        start_idx = self.get_path_start(curr_idx)
                        end_idx = self.get_path_end(curr_idx)
                        pathlen = self.get_pathlen(curr_idx)

                        if start_idx < end_idx:
                            self.buffers[self.PATH_START_KEY][start_idx:end_idx + 1] = INVALID_IDX
                            self._free_sample_buffers(list(range(start_idx, end_idx + 1)))
                        else:
                            self.buffers[self.PATH_START_KEY][start_idx:self.buffer_size] = INVALID_IDX
                            self.buffers[self.PATH_START_KEY][0:end_idx + 1] = INVALID_IDX
                            self._free_sample_buffers(list(range(start_idx, self.buffer_size)))
                            self._free_sample_buffers(list(range(0, end_idx + 1)))
                        self.num_paths -= 1
                        i += pathlen + 1
                        self.buffer_tail = (end_idx + 1) % self.buffer_size
                    else:
                        i += 1
        else:
            self.buffer_tail = idx[0]
        return

    def _store_path(self, path, idx):
        n = path.pathlength()
        for key, data in self.buffers.items():
            if key != self.PATH_START_KEY and key != self.PATH_END_KEY and key != self.TERMINATE_KEY:
                val = getattr(path, key)
                val_len = len(val)
                assert val_len == n or val_len == n + 1
                data[idx[:val_len]] = val

        self.buffers[self.TERMINATE_KEY][idx] = path.terminate.value
        self.buffers[self.PATH_START_KEY][idx] = idx[0]
        self.buffers[self.PATH_END_KEY][idx] = idx[-1]
        return


class SampleBuffer(object):
    def __init__(self, size):
        self.idx_to_slot = np.empty(shape=[size], dtype=int)
        self.slot_to_idx = np.empty(shape=[size], dtype=int)
        self.count = 0
        self.clear()
        return

    def clear(self):
        self.idx_to_slot.fill(INVALID_IDX)
        self.slot_to_idx.fill(INVALID_IDX)
        self.count = 0
        return

    def is_valid(self, idx):
        return self.idx_to_slot[idx] != INVALID_IDX

    def get_size(self):
        return self.idx_to_slot.shape[0]

    def add(self, idx):
        for i in idx:
            if not self.is_valid(i):
                new_slot = self.count
                assert new_slot >= 0

                self.idx_to_slot[i] = new_slot
                self.slot_to_idx[new_slot] = i
                self.count += 1
        return

    def free(self, idx):
        for i in idx:
            if self.is_valid(i):
                slot = self.idx_to_slot[i]
                last_slot = self.count - 1
                last_idx = self.slot_to_idx[last_slot]

                self.idx_to_slot[last_idx] = slot
                self.slot_to_idx[slot] = last_idx
                self.idx_to_slot[i] = INVALID_IDX
                self.slot_to_idx[last_slot] = INVALID_IDX
                self.count -= 1
        return

    def sample(self, n):
        if self.count > 0:
            slots = np.random.randint(0, self.count, size=n)
            idx = self.slot_to_idx[slots]
        else:
            idx = np.empty(shape=[0], dtype=int)
        return idx

    def check_consistency(self):
        valid = True
        if self.count < 0:
            valid = False

        if valid:
            for i in range(self.get_size()):
                if self.is_valid(i):
                    s = self.idx_to_slot[i]
                    if self.slot_to_idx[s] != i:
                        valid = False
                        break

                s2i = self.slot_to_idx[i]
                if s2i != INVALID_IDX:
                    i2s = self.idx_to_slot[s2i]
                    if i2s != i:
                        valid = False
                        break

        count0 = np.sum(self.idx_to_slot == INVALID_IDX)
        count1 = np.sum(self.slot_to_idx == INVALID_IDX)
        valid &= count0 == count1
        return valid


def calc_logp_gaussian(x_tf, mean_tf, std_tf):
    dim = tf.to_float(tf.shape(x_tf)[-1])

    if mean_tf is None:
        diff_tf = x_tf
    else:
        diff_tf = x_tf - mean_tf

    logp_tf = -0.5 * tf.reduce_sum(tf.square(diff_tf / std_tf), axis=-1)
    logp_tf += -0.5 * dim * np.log(2 * np.pi) - \
        tf.reduce_sum(tf.log(std_tf), axis=-1)

    return logp_tf


def fc_net(input, layers_sizes, activation, reuse=None, flatten=False):  # build fully connected network
    curr_tf = input
    for i, size in enumerate(layers_sizes):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                      units=size,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=activation if i < len(layers_sizes) - 1 else None)
    if flatten:
        assert layers_sizes[-1] == 1
        curr_tf = tf.reshape(curr_tf, [-1])

    return curr_tf


def calc_bound_loss(x_tf, bound_min, bound_max):
    # penalty for violating bounds
    # 增加了一个对超出界限的惩罚
    violation_min = tf.minimum(x_tf - bound_min, 0)
    violation_max = tf.maximum(x_tf - bound_max, 0)
    violation = tf.reduce_sum(tf.square(violation_min), axis=-1) + tf.reduce_sum(tf.square(violation_max), axis=-1)
    loss = 0.5 * tf.reduce_mean(violation)
    return loss


def compute_return(rewards, gamma, td_lambda, val_t):
    # computes td-lambda return of path
    path_len = len(rewards)
    assert len(val_t) == path_len + 1

    return_t = np.zeros(path_len)
    last_val = rewards[-1] + gamma * val_t[-1]
    return_t[-1] = last_val

    for i in reversed(range(0, path_len - 1)):
        curr_r = rewards[i]
        next_ret = return_t[i + 1]
        curr_val = curr_r + gamma * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
        return_t[i] = curr_val
    return return_t


def lerp(x, y, t):
    return (1 - t) * x + t * y


def log_lerp(x, y, t):
    return np.exp(lerp(np.log(x), np.log(y), t))


def flatten(arr_list):
    return np.concatenate([np.reshape(a, [-1]) for a in arr_list], axis=0)


def flip_coin(p):
    rand_num = np.random.binomial(1, p, 1)
    return rand_num[0] == 1


class ExpParams(object):

    def __init__(self, config=dict()):
        self.rate = config.get("Rate", 0.2)
        self.init_action_rate = config.get("InitActionRate", 0)
        self.noise = config.get("Noise", 0.1)
        self.noise_internal = config.get("NoiseInternal", 0)
        self.temp = config.get("Temp", 0.1)
        return

    def __str__(self):
        str = ''
        str += 'Rate: {:.2f}\n'.format(self.rate)
        str += 'InitActionRate: {:.2f}\n'.format(self.init_action_rate)
        str += 'Noise: {:.2f}\n'.format(self.noise)
        str += 'NoiseInternal: {:.2f}\n'.format(self.noise_internal)
        str += 'Temp: {:.2f}\n'.format(self.temp)
        return str

    def lerp(self, other, t):
        lerp_params = ExpParams()
        lerp_params.rate = lerp(self.rate, other.rate, t)
        lerp_params.init_action_rate = lerp(self.init_action_rate, other.init_action_rate, t)
        lerp_params.noise = lerp(self.noise, other.noise, t)
        lerp_params.noise_internal = lerp(self.noise_internal, other.noise_internal, t)
        lerp_params.temp = log_lerp(self.temp, other.temp, t)
        return lerp_params


# class ActionSpace(Enum):
#     Null = 0
#     Continuous = 1
#     Discrete = 2


class PPOAgent(object):

    class Mode(Enum):
        TRAIN = 0
        TEST = 1
        TRAIN_END = 2

    # UPDATE_PERIOD_KEY = "UpdatePeriod"
    # ITERS_PER_UPDATE = "ItersPerUpdate"
    # DISCOUNT_KEY = "Discount"
    # MINI_BATCH_SIZE_KEY = "MiniBatchSize"
    # REPLAY_BUFFER_SIZE_KEY = "ReplayBufferSize"
    # INIT_SAMPLES_KEY = "InitSamples"
    # NORMALIZER_SAMPLES_KEY = "NormalizerSamples"

    # OUTPUT_ITERS_KEY = "OutputIters"
    # INT_OUTPUT_ITERS_KEY = "IntOutputIters"
    # TEST_EPISODES_KEY = "TestEpisodes"

    # EXP_ANNEAL_SAMPLES_KEY = "ExpAnnealSamples"
    # EXP_PARAM_BEG_KEY = "ExpParamsBeg"
    # EXP_PARAM_END_KEY = "ExpParamsEnd"

    # NAME = "PPO"
    # EPOCHS_KEY = "Epochs"
    # BATCH_SIZE_KEY = "BatchSize"
    # RATIO_CLIP_KEY = "RatioClip"
    # NORM_ADV_CLIP_KEY = "NormAdvClip"
    # TD_LAMBDA_KEY = "TDLambda"
    # TAR_CLIP_FRAC = "TarClipFrac"
    # ACTOR_STEPSIZE_DECAY = "ActorStepsizeDecay"

    # ACTOR_NET_KEY = 'ActorNet'
    # ACTOR_STEPSIZE_KEY = 'ActorStepsize'
    # ACTOR_MOMENTUM_KEY = 'ActorMomentum'
    # ACTOR_WEIGHT_DECAY_KEY = 'ActorWeightDecay'
    # ACTOR_INIT_OUTPUT_SCALE_KEY = 'ActorInitOutputScale'

    # CRITIC_NET_KEY = 'CriticNet'
    # CRITIC_STEPSIZE_KEY = 'CriticStepsize'
    # CRITIC_MOMENTUM_KEY = 'CriticMomentum'
    # CRITIC_WEIGHT_DECAY_KEY = 'CriticWeightDecay'
    EXP_ACTION_FLAG = 1 << 0

    # RESOURCE_SCOPE = 'resource'
    # SOLVER_SCOPE = 'solvers'

    def __init__(self, env, id, config):
        self.tf_scope = 'agent'
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        self.env = env
        self.id = id
        self.logger = Logger()
        self._mode = self.Mode.TRAIN

        assert self._check_action_space(), \
            Logger.print("Invalid action space, got {:s}".format(str(self.get_action_space())))

        self._enable_training = True
        self.path = Path()
        self.iter = int(0)
        self.start_time = time.time()
        self._update_counter = 0

        self.update_period = config.get("UpdatePeriod", 1)  # simulated time (seconds) before each training update
        self.iters_per_update = config.get("ItersPerUpdate", 1)
        self.discount = config.get("Discount", 0.95)
        self.mini_batch_size = config.get("MiniBatchSize", 32)
        self.replay_buffer_size = config.get("ReplayBufferSize", 50000)
        self.init_samples = config.get("InitSamples", 1000)
        self.normalizer_samples = config.get("NormalizerSamples", np.inf)
        self._local_mini_batch_size = self.mini_batch_size  # batch size for each work for multiprocessing

        num_procs = MPIUtil.get_num_procs()
        self._local_mini_batch_size = int(np.ceil(self.mini_batch_size / num_procs))
        self._local_mini_batch_size = np.maximum(self._local_mini_batch_size, 1)
        self.mini_batch_size = self._local_mini_batch_size * num_procs

        self._need_normalizer_update = self.normalizer_samples > 0
        self._total_sample_count = 0

        self._output_dir = ""
        self._int_output_dir = ""
        self.output_iters = config.get("OutputIters", 100)
        self.int_output_iters = config.get("IntOutputIters", 100)

        self.train_return = 0.0
        self.test_episodes = config.get("TestEpisodes", 0)
        self.test_episode_count = int(0)
        self.test_return = 0.0
        self.avg_test_return = 0.0

        self.obs = self.env.reset()
        self.exp_anneal_samples = config.get("ExpAnnealSamples", 320000)

        self.exp_params_beg = ExpParams(
            config.get("ExpParamsBeg", {"Rate": 1, "InitActionRate": 1, "Noise": 0.05, "NoiseInternal": 0, "Temp": 0.1}))
        self.exp_params_end = ExpParams(
            config.get("ExpParamsEnd", {"Rate": 0.2, "InitActionRate": 0.01, "Noise": 0.05, "NoiseInternal": 0, "Temp": 0.001}))
        self.exp_params_curr = ExpParams()

        assert(self.exp_params_beg.noise == self.exp_params_end.noise)  # noise std should not change

        self.exp_params_curr = copy.deepcopy(self.exp_params_beg)

        self.val_min, self.val_max = self._calc_val_bounds(self.discount)
        self.val_fail, self.val_succ = self._calc_term_vals(self.discount)

        self.epochs = config.get("Epochs", 1)
        self.batch_size = config.get("BatchSize", 1024)
        self.ratio_clip = config.get("RatioClip", 0.2)
        self.norm_adv_clip = config.get("NormAdvClip", 5)
        self.td_lambda = config.get("TDLambda", 0.95)
        self.tar_clip_frac = config.get("TarClipFrac", -1)
        self.actor_stepsize_decay = config.get("ActorStepsizeDecay", 0.5)

        local_batch_size = int(self.batch_size / num_procs)
        min_replay_size = 2 * local_batch_size  # needed to prevent buffer overflow
        assert(self.replay_buffer_size > min_replay_size)

        self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)

        # Graph Params

        self.actor_net_name = config.get("ActorNet", "fc_2layers_1024units")
        self.critic_net_name = config.get("CriticNet", "fc_2layers_1024units")
        self.actor_init_output_scale = config.get("ActorInitOutputScale", 1)

        # Losses Params

        self.actor_weight_decay = config.get("ActorWeightDecay", 0.0)
        self.critic_weight_decay = config.get("CriticWeightDecay", 0.0)

        # Solver Params

        self.actor_stepsize = config.get("ActorStepsize", 0.001)
        self.actor_momentum = config.get("CriticStepsize", 0.9)
        self.critic_stepsize = config.get("CriticStepsize", 0.01)
        self.critic_momentum = config.get("CriticMomentum", 0.9)

        self._build_replay_buffer(self.replay_buffer_size)
        self._build_normalizers()
        self._build_bounds()
        self.reset()

        self._build_graph()
        self._init_normalizers()
        self._exp_action = False

        return

    def __del__(self):
        self.sess.close()
        return

    def __str__(self):
        action_space_str = str(self.get_action_space())
        info_str = ""
        info_str += '"ID": {:d},\n "Type": "{:s}",\n "ActionSpace": "{:s}",\n "StateDim": {:d},\n "GoalDim": {:d},\n "ActionDim": {:d}'.format(
            self.id, self.NAME, action_space_str[action_space_str.rfind('.') + 1:], self.get_state_size(), self.get_goal_size(), self.get_action_size())
        return "{\n" + info_str + "\n}"

    def reset(self):
        self.path.clear()
        self._exp_action = False
        return

    def _check_action_space(self):
        action_space = self.get_action_space()
        print(action_space)
        print("----------------------------")
        print(action_space == ActionSpace.Continuous)
        return action_space == ActionSpace.Continuous

    def get_output_dir(self):
        return self._output_dir

    def set_output_dir(self, out_dir):
        self._output_dir = out_dir
        if (self._output_dir != ""):
            self.logger.configure_output_file(out_dir + "/agent" + str(self.id) + "_log.txt")
        return

    output_dir = property(get_output_dir, set_output_dir)

    def get_int_output_dir(self):
        return self._int_output_dir

    def set_int_output_dir(self, out_dir):
        self._int_output_dir = out_dir
        return

    int_output_dir = property(get_int_output_dir, set_int_output_dir)

    def get_enable_training(self):
        return self._enable_training

    def set_enable_training(self, enable):
        self._enable_training = enable
        if (self._enable_training):
            self.reset()
        return

    enable_training = property(get_enable_training, set_enable_training)

    def enable_testing(self):
        return self.test_episodes > 0

    def _build_nets(self):

        s_size = self.get_state_size()
        g_size = self.get_goal_size()
        a_size = self.get_action_size()

        # setup input tensors
        self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
        self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self.tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
        self.g_tf = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g")
        self.old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")
        # 没太了解这个exp_mask的作用
        self.exp_mask_tf = tf.placeholder(tf.float32, shape=[None], name="exp_mask")

        with tf.variable_scope('main'):
            with tf.variable_scope('actor'):
                self.a_mean_tf = self._build_net_actor(self.actor_net_name, self.actor_init_output_scale)
            with tf.variable_scope('critic'):
                self.critic_tf = self._build_net_critic(self.critic_net_name)

        if (self.a_mean_tf is not None):
            Logger.print('Built actor net: ' + self.actor_net_name)

        if (self.critic_tf is not None):
            Logger.print('Built critic net: ' + self.critic_net_name)

        # norm这个似乎是给action加入噪声
        self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)
        norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(shape=tf.shape(self.a_mean_tf))
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
        self.sample_a_logp_tf = calc_logp_gaussian(x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf)

        return

    def _build_net_actor(self, net_name, init_output_scale):
        # 这个地方nomoralize的作用是把输入的东西全部都均一化了，并且还要能够实现反均一化
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]

        h = self.build_net(input_tfs)
        norm_a_tf = tf.layers.dense(inputs=h, units=self.get_action_size(), activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale))

        a_tf = self.a_norm.unnormalize_tf(norm_a_tf)
        return a_tf

    def _build_net_critic(self, net_name):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]

        h = self.build_net(input_tfs)
        norm_val_tf = tf.layers.dense(inputs=h, units=1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        norm_val_tf = tf.reshape(norm_val_tf, [-1])
        val_tf = self.val_norm.unnormalize_tf(norm_val_tf)
        return val_tf

    def build_net(self, input_tfs, reuse=False):
        layers = [1024, 512]
        activation = tf.nn.relu

        input_tf = tf.concat(axis=-1, values=input_tfs)
        h = fc_net(input_tf, layers, activation=activation, reuse=reuse)
        h = activation(h)
        return h

    def _build_losses(self):
        # critic loss, 就是简单的二次误差那种形式
        # 这个地方的loss跟torch_ac的是不一样的
        norm_val_diff = self.val_norm.normalize_tf(self.tar_val_tf) - self.val_norm.normalize_tf(self.critic_tf)
        self.critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

        if (self.critic_weight_decay != 0):
            self.critic_loss_tf += self.critic_weight_decay * self._weight_decay_loss('main/critic')

        norm_tar_a_tf = self.a_norm.normalize_tf(self.a_tf)
        self._norm_a_mean_tf = self.a_norm.normalize_tf(self.a_mean_tf)

        self.logp_tf = calc_logp_gaussian(norm_tar_a_tf, self._norm_a_mean_tf, self.norm_a_std_tf)
        ratio_tf = tf.exp(self.logp_tf - self.old_logp_tf)
        actor_loss0 = self.adv_tf * ratio_tf
        actor_loss1 = self.adv_tf * tf.clip_by_value(ratio_tf, 1.0 - self.ratio_clip, 1 + self.ratio_clip)
        self.actor_loss_tf = -tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))

        # 这个bound loss是在之前没见过的啊
        # 可能是deepmimic里面独特的
        norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
        norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
        a_bound_loss = calc_bound_loss(self._norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max)
        self.actor_loss_tf += a_bound_loss

        if (self.actor_weight_decay != 0):
            self.actor_loss_tf += self.actor_weight_decay * self._weight_decay_loss('main/actor')

        # for debugging
        self.clip_frac_tf = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip)))

        return

    def _build_solvers(self):
        critic_vars = self._tf_vars('main/critic')
        critic_opt = tf.train.MomentumOptimizer(learning_rate=self.critic_stepsize, momentum=self.critic_momentum)
        self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
        self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

        self._actor_stepsize_tf = tf.get_variable(dtype=tf.float32, name='actor_stepsize', initializer=self.actor_stepsize, trainable=False)
        self._actor_stepsize_ph = tf.get_variable(dtype=tf.float32, name='actor_stepsize_ph', shape=[])
        self._actor_stepsize_update_op = self._actor_stepsize_tf.assign(self._actor_stepsize_ph)

        actor_vars = self._tf_vars('main/actor')
        actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize_tf, momentum=self.actor_momentum)
        self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
        self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

        return

    def _decide_action(self, s, g):
        with self.sess.as_default(), self.graph.as_default():
            self._exp_action = self._enable_stoch_policy() and flip_coin(self.exp_params_curr.rate)
            a, logp = self._eval_actor(s, g, self._exp_action)
        return a[0], logp[0]

    def _eval_actor(self, s, g, enable_exp):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.exp_mask_tf: np.array([1 if enable_exp else 0])
        }

        a, logp = self.sess.run([self.sample_a_tf, self.sample_a_logp_tf], feed_dict=feed)
        return a, logp

    def _eval_critic(self, s, g):
        with self.sess.as_default(), self.graph.as_default():
            s = np.reshape(s, [-1, self.get_state_size()])
            g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

            feed = {
                self.s_tf: s,
                self.g_tf: g
            }
            val = self.critic_tf.eval(feed)
        return val

    def collect_experience(self):
        # print("collect experience")

        time_consumed = 0.0
        done = False
        while (self.update_period > time_consumed):
            self.path.states.append(self.obs)
            self.path.goals.append(np.zeros((0,), dtype=np.float64))
            a, logp = self._decide_action(s=self.obs, g=np.zeros((0,), dtype=np.float64))
            assert len(np.shape(a)) == 1
            assert len(np.shape(logp)) <= 1
            self.obs, reward, done, info = self.env.step(a)
            time_consumed += info['time_elapsed']
            if info['valid'] != 1.0:
                self.reset()
                self.obs = self.env.reset()
                continue
            elif info['valid'] == 1.0 and done is True:

                self.end_episode()
                # self.env.reset()
                self.obs = self.env.reset()
                self.reset()
                continue
            else:
                pass

            flags = self._record_flags()

            self.path.rewards.append(reward)
            self.path.actions.append(a)
            self.path.logps.append(logp)
            self.path.flags.append(flags)

        # print("tiem consumed", time_consumed)

    def train(self):
        # print("train")
        if (self._mode == self.Mode.TRAIN and self.enable_training):
            # self._update_counter += timestep
            # while self._update_counter >= self.update_period:
            # print("train")
            with self.sess.as_default(), self.graph.as_default():
                self._train()
            self._update_exp_params()
            self.env.set_sample_count(self._total_sample_count)
            # self._update_counter -= self.update_period
        return

    def _train(self):
        # print("_train")
        samples = self.replay_buffer.total_count
        self._total_sample_count = int(MPIUtil.reduce_sum(samples))
        end_training = False

        if (self.replay_buffer_initialized):
            if (self._valid_train_step()):
                prev_iter = self.iter
                iters = self._get_iters_per_update()
                avg_train_return = MPIUtil.reduce_avg(self.train_return)

                for i in range(iters):
                    curr_iter = self.iter
                    wall_time = time.time() - self.start_time
                    wall_time /= 60 * 60  # store time in hours

                    has_goal = self.has_goal()
                    s_mean = np.mean(self.s_norm.mean)
                    s_std = np.mean(self.s_norm.std)
                    g_mean = np.mean(self.g_norm.mean) if has_goal else 0
                    g_std = np.mean(self.g_norm.std) if has_goal else 0

                    self.logger.log_tabular("Iteration", self.iter)
                    self.logger.log_tabular("Wall_Time", wall_time)
                    self.logger.log_tabular("Samples", self._total_sample_count)
                    self.logger.log_tabular("Train_Return", avg_train_return)
                    self.logger.log_tabular("Test_Return", self.avg_test_return)
                    self.logger.log_tabular("State_Mean", s_mean)
                    self.logger.log_tabular("State_Std", s_std)
                    self.logger.log_tabular("Goal_Mean", g_mean)
                    self.logger.log_tabular("Goal_Std", g_std)
                    self.logger.log_tabular("Exp_Rate", self.exp_params_curr.rate)
                    self.logger.log_tabular("Exp_Noise", self.exp_params_curr.noise)
                    self.logger.log_tabular("Exp_Temp", self.exp_params_curr.temp)

                    self._update_iter(self.iter + 1)
                    self._train_step()

                    Logger.print("Agent " + str(self.id))
                    self.logger.print_tabular()
                    Logger.print("")

                    if (self._enable_output() and curr_iter % self.int_output_iters == 0):
                        self.logger.dump_tabular()

                if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
                    end_training = self.enable_testing()

        else:

            Logger.print("Agent " + str(self.id))
            Logger.print("Samples: " + str(self._total_sample_count))
            Logger.print("")

            if (self._total_sample_count >= self.init_samples):
                self.replay_buffer_initialized = True
                end_training = self.enable_testing()

        if self._need_normalizer_update:
            self._update_normalizers()
            self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

        if end_training:
            self._init_mode_train_end()

        return

    def _train_step(self):
        adv_eps = 1e-5

        start_idx = self.replay_buffer.buffer_tail
        end_idx = self.replay_buffer.buffer_head
        assert(start_idx == 0)
        assert(self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size)  # must avoid overflow
        assert(start_idx < end_idx)

        idx = np.array(list(range(start_idx, end_idx)))
        end_mask = self.replay_buffer.is_path_end(idx)
        end_mask = np.logical_not(end_mask)

        vals = self._compute_batch_vals(start_idx, end_idx)
        new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)

        valid_idx = idx[end_mask]
        exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
        num_valid_idx = valid_idx.shape[0]
        num_exp_idx = exp_idx.shape[0]
        exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])

        local_sample_count = valid_idx.size
        global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
        mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

        adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]
        new_vals = np.clip(new_vals, self.val_min, self.val_max)

        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        adv = (adv - adv_mean) / (adv_std + adv_eps)
        adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)

        critic_loss = 0
        actor_loss = 0
        actor_clip_frac = 0

        for e in range(self.epochs):
            np.random.shuffle(valid_idx)
            np.random.shuffle(exp_idx)

            for b in range(mini_batches):
                batch_idx_beg = b * self._local_mini_batch_size
                batch_idx_end = batch_idx_beg + self._local_mini_batch_size

                critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
                actor_batch = critic_batch.copy()
                critic_batch = np.mod(critic_batch, num_valid_idx)
                actor_batch = np.mod(actor_batch, num_exp_idx)
                shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)

                critic_batch = valid_idx[critic_batch]
                actor_batch = exp_idx[actor_batch]
                critic_batch_vals = new_vals[critic_batch]
                actor_batch_adv = adv[actor_batch[:, 1]]

                critic_s = self.replay_buffer.get('states', critic_batch)
                critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
                curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

                actor_s = self.replay_buffer.get("states", actor_batch[:, 0])
                actor_g = self.replay_buffer.get("goals", actor_batch[:, 0]) if self.has_goal() else None
                actor_a = self.replay_buffer.get("actions", actor_batch[:, 0])
                actor_logp = self.replay_buffer.get("logps", actor_batch[:, 0])
                curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a, actor_logp, actor_batch_adv)

                critic_loss += curr_critic_loss
                actor_loss += np.abs(curr_actor_loss)
                actor_clip_frac += curr_actor_clip_frac

                if (shuffle_actor):
                    np.random.shuffle(exp_idx)

        total_batches = mini_batches * self.epochs
        critic_loss /= total_batches
        actor_loss /= total_batches
        actor_clip_frac /= total_batches

        critic_loss = MPIUtil.reduce_avg(critic_loss)
        actor_loss = MPIUtil.reduce_avg(actor_loss)
        actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)

        critic_stepsize = self.critic_solver.get_stepsize()
        actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

        self.logger.log_tabular('Critic_Loss', critic_loss)
        self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
        self.logger.log_tabular('Actor_Loss', actor_loss)
        self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
        self.logger.log_tabular('Clip_Frac', actor_clip_frac)
        self.logger.log_tabular('Adv_Mean', adv_mean)
        self.logger.log_tabular('Adv_Std', adv_std)

        self.replay_buffer.clear()

        return

    def _get_iters_per_update(self):
        return 1

    def _valid_train_step(self):
        samples = self.replay_buffer.get_current_size()
        exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
        global_sample_count = int(MPIUtil.reduce_sum(samples))
        global_exp_min = int(MPIUtil.reduce_min(exp_samples))
        return (global_sample_count > self.batch_size) and (global_exp_min > 0)

    def _compute_batch_vals(self, start_idx, end_idx):
        states = self.replay_buffer.get_all("states")[start_idx:end_idx]
        goals = self.replay_buffer.get_all("goals")[start_idx:end_idx] if self.has_goal() else None
        idx = np.array(list(range(start_idx, end_idx)))
        is_end = self.replay_buffer.is_path_end(idx)
        is_fail = self.replay_buffer.check_terminal_flag(idx, Terminate.Fail)
        is_succ = self.replay_buffer.check_terminal_flag(idx, Terminate.Succ)
        is_fail = np.logical_and(is_end, is_fail)
        is_succ = np.logical_and(is_end, is_succ)

        vals = self._eval_critic(states, goals)
        vals[is_fail] = self.val_fail
        vals[is_succ] = self.val_succ

        return vals

    def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer):
        rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]

        if self.discount == 0:
            new_vals = rewards.copy()
        else:
            new_vals = np.zeros_like(val_buffer)

            curr_idx = start_idx
            while curr_idx < end_idx:
                idx0 = curr_idx - start_idx
                idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
                r = rewards[idx0:idx1]
                v = val_buffer[idx0:(idx1 + 1)]

                new_vals[idx0:idx1] = compute_return(r, self.discount, self.td_lambda, v)
                curr_idx = idx1 + start_idx + 1

        return new_vals

    def _update_critic(self, s, g, tar_vals):
        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.tar_val_tf: tar_vals
        }

        loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
        self.critic_solver.update(grads)
        return loss

    def _update_actor(self, s, g, a, logp, adv):
        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.a_tf: a,
            self.adv_tf: adv,
            self.old_logp_tf: logp
        }

        loss, grads, clip_frac = self.sess.run([self.actor_loss_tf, self.actor_grad_tf, self.clip_frac_tf], feed)
        self.actor_solver.update(grads)

        return loss, clip_frac

    def update_actor_stepsize(self, clip_frac):
        clip_tol = 1.5
        # step_scale = 2
        max_stepsize = 1e-2
        min_stepsize = 1e-8
        warmup_iters = 5

        actor_stepsize = self.actor_solver.get_stepsize()
        if (self.tar_clip_frac >= 0 and self.iter > warmup_iters):
            min_clip = self.tar_clip_frac / clip_tol
            max_clip = self.tar_clip_frac * clip_tol
            under_tol = clip_frac < min_clip
            over_tol = clip_frac > max_clip

            if (over_tol or under_tol):
                if (over_tol):
                    actor_stepsize *= self.actor_stepsize_decay
                else:
                    actor_stepsize /= self.actor_stepsize_decay

                actor_stepsize = np.clip(actor_stepsize, min_stepsize, max_stepsize)
                self.set_actor_stepsize(actor_stepsize)

        return actor_stepsize

    def set_actor_stepsize(self, stepsize):
        feed = {
            self._actor_stepsize_ph: stepsize,
        }
        self.sess.run(self._actor_stepsize_update_op, feed)
        return

    def _build_normalizers(self):
        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
            with tf.variable_scope("resource"):
                # 为什么要把sess传进去呢
                self.s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(), self.build_state_norm_groups(self.id))
                self.s_norm.set_mean_std(-self.build_state_offset(self.id), 1 / self.build_state_scale(self.id))

                self.g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(), self.build_goal_norm_groups(self.id))
                self.g_norm.set_mean_std(-self.build_goal_offset(self.id), 1 / self.build_goal_scale(self.id))

                self.a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
                self.a_norm.set_mean_std(-self.build_action_offset(self.id), 1 / self.build_action_scale(self.id))

                val_offset, val_scale = self._calc_val_offset_scale(self.discount)
                self.val_norm = TFNormalizer(self.sess, 'val_norm', 1)
                self.val_norm.set_mean_std(-val_offset, 1.0 / val_scale)
        return

    def _init_normalizers(self):
        # normalizer是干什么的？
        with self.sess.as_default(), self.graph.as_default():
            self.s_norm.update()
            self.g_norm.update()
            self.a_norm.update()
            self.val_norm.update()
        return

    def _load_normalizers(self):
        self.s_norm.load()
        self.g_norm.load()
        self.a_norm.load()
        self.val_norm.load()
        return

    def _record_normalizers(self, path):
        states = np.array(path.states)
        self.s_norm.record(states)

        if self.has_goal():
            goals = np.array(path.goals)
            self.g_norm.record(goals)
        return

    def _update_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            self.s_norm.update()

            if self.has_goal():
                self.g_norm.update()
        return

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())
        self._sync_solvers()
        return

    def _sync_solvers(self):
        self.actor_solver.sync()
        self.critic_solver.sync()
        return

    def _enable_stoch_policy(self):
        # 训练的时候policy是需要stoch的
        return self.enable_training and (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END)

    def _record_flags(self):
        flags = int(0)
        if (self._exp_action):
            flags = flags | self.EXP_ACTION_FLAG
        return flags

    def _calc_updated_vals(self, idx):
        r = self.replay_buffer.get('rewards', idx)

        if self.discount == 0:
            new_V = r
        else:
            next_idx = self.replay_buffer.get_next_idx(idx)
            s_next = self.replay_buffer.get('states', next_idx)
            g_next = self.replay_buffer.get('goals', next_idx) if self.has_goal() else None

            is_end = self.replay_buffer.is_path_end(idx)
            is_fail = self.replay_buffer.check_terminal_flag(idx, Terminate.Fail)
            is_succ = self.replay_buffer.check_terminal_flag(idx, Terminate.Succ)
            is_fail = np.logical_and(is_end, is_fail)
            is_succ = np.logical_and(is_end, is_succ)

            V_next = self._eval_critic(s_next, g_next)
            V_next[is_fail] = self.val_fail
            V_next[is_succ] = self.val_succ

            new_V = r + self.discount * V_next
        return new_V

    def _calc_action_logp(self, norm_action_deltas):
        # norm action delta are for the normalized actions (scaled by self.a_norm.std)
        stdev = self.exp_params_curr.noise
        assert stdev > 0

        a_size = self.get_action_size()
        logp = -0.5 / (stdev * stdev) * np.sum(np.square(norm_action_deltas), axis=-1)
        logp += -0.5 * a_size * np.log(2 * np.pi)
        logp += -a_size * np.log(stdev)
        return logp

    def _log_val(self, s, g):
        val = self._eval_critic(s, g)
        norm_val = self.val_norm.normalize(val)
        # --**--
        # 没懂是干啥的？再查找一下吧
        self.env.log_val(self.id, norm_val[0])
        return

    def _build_replay_buffer(self, buffer_size):
        num_procs = MPIUtil.get_num_procs()
        buffer_size = int(buffer_size / num_procs)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.replay_buffer_initialized = False
        self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
        return

    def save_model(self, out_path):
        with self.sess.as_default(), self.graph.as_default():
            try:
                save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
                Logger.print('Model saved to: ' + save_path)
            except:
                Logger.print("Failed to save model to: " + save_path)
        return

    # def jy_save_model(self, out_path):
    #     with self.sess.as_default(), self.graph.as_default():
    #         try:
    #             save_path = self.saver.save(self.sess, out_path, write_meta_graph=True, write_state=False)
    #             Logger.print('Model saved to: ' + save_path)
    #         except:
    #             Logger.print("Failed to save model to: " + save_path)
    #     return

    def load_model(self, in_path):
        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            self._load_normalizers()
            Logger.print('Model loaded from: ' + in_path)
        return

    def _get_output_path(self):
        assert(self.output_dir != '')
        file_path = self.output_dir + '/agent' + str(self.id) + '_model.ckpt'
        return file_path

    def _get_int_output_path(self):
        assert(self.int_output_dir != '')
        file_path = self.int_output_dir + ('/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
        return file_path

    def _build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_nets()
                with tf.variable_scope("solvers"):
                    self._build_losses()
                    self._build_solvers()

                self._initialize_vars()
                self._build_saver()
        return

    def build_action_offset(self, agent_id):
        # from cCtController::BuildActionOffsetScale
        res = np.zeros((36,))
        res[3] = -0.2
        res[7] = -0.2
        res[11] = -0.2
        res[12] = 1.57
        res[16] = -0.2
        res[20] = -0.2
        res[21] = -1.57
        res[25] = -0.2
        res[26] = 1.57
        res[30] = -0.2
        res[34] = -0.2
        res[35] = -1.57
        return res

    def build_action_scale(self, agent_id):
        res = np.ones((36,))
        res[0] = 0.20833333333333334
        res[4] = 0.25
        res[8] = 0.12077294685990339
        res[12] = 0.1592356687898089
        res[13] = 0.1592356687898089
        res[17] = 0.07961783439490445
        res[21] = 0.1592356687898089
        res[22] = 0.12077294685990339
        res[26] = 0.1592356687898089
        res[27] = 0.1592356687898089
        res[31] = 0.10775862068965517
        res[35] = 0.1592356687898089
        return res

    def build_action_bound_min(self):
        res = np.ones((36,), dtype=np.float64)
        res = res * -1
        res[0] = -4.8
        res[4] = -4.0
        res[8] = -7.7799999999999999
        res[12] = -7.8500000000000005
        res[13] = -6.28
        res[17] = -12.56
        res[21] = -4.71
        res[22] = -7.7799999999999999
        res[26] = -7.8500000000000005
        res[27] = -6.28
        res[31] = -8.46
        res[35] = -4.71
        return res

    def build_action_bound_max(self):
        res = np.ones((36,))
        res[0] = 4.8
        res[4] = 4.0
        res[8] = 8.78
        res[12] = 4.71
        res[13] = 6.28
        res[17] = 12.56
        res[21] = 7.8500000000000005
        res[22] = 8.78
        res[26] = 4.71
        res[27] = 6.28
        res[31] = 10.100000000000001
        res[35] = 7.8500000000000005
        return res

    def build_state_scale(self, agent_id):
        res = np.ones((197,), dtype=np.float64)
        res[0] = 2
        return res

    def build_state_offset(self, agent_id):
        res = np.zeros((197,), dtype=np.float64)
        res[0] = -0.5
        return res

    def build_state_norm_groups(self, agent_id):
        # 不知道这个的作用是什么
        res = np.zeros((197,), dtype=np.float64)
        res[0] = -1
        return res

    def build_goal_scale(self, agent_id):
        # print(np.array(self._core.BuildGoalScale(agent_id)))
        return np.array([])

    def build_goal_offset(self, agent_id):
        return np.array([])

    def build_goal_norm_groups(self, agent_id):
        return np.array([])

    def has_goal(self):
        # goalsize目测始终是0啊？
        return self.get_goal_size() > 0

    def get_name(self):
        return self.NAME

    def get_action_space(self):
        print("action space!!!!!!!!", self.env.get_action_space(self.id))
        return self.env.get_action_space(self.id)

    def get_state_size(self):
        return 197

    def get_goal_size(self):
        return 0

    def get_action_size(self):
        return 36

    def get_reward_max(self, agent_id):
        return 1.0

    def get_reward_min(self, agent_id):
        return 0.0

    def get_reward_fail(self, agent_id):
        return 0.0

    def get_reward_succ(self, agent_id):
        return 1.0

    def get_num_actions(self):
        print("num actions!!!!!!!!", self.env.get_num_actions(self.id))
        return self.env.get_num_actions(self.id)

    def _calc_val_bounds(self, discount):
        r_min = self.get_reward_min(self.id)
        r_max = self.get_reward_max(self.id)
        assert(r_min <= r_max)

        val_min = r_min / (1.0 - discount)
        val_max = r_max / (1.0 - discount)
        return val_min, val_max

    def _calc_val_offset_scale(self, discount):
        val_min, val_max = self._calc_val_bounds(discount)
        val_offset = 0
        val_scale = 1

        if (np.isfinite(val_min) and np.isfinite(val_max)):
            val_offset = -0.5 * (val_max + val_min)
            val_scale = 2 / (val_max - val_min)

        return val_offset, val_scale

    def _calc_term_vals(self, discount):
        r_fail = self.get_reward_fail(self.id)
        r_succ = self.get_reward_succ(self.id)

        r_min = self.get_reward_min(self.id)
        r_max = self.get_reward_max(self.id)
        assert(r_fail <= r_max and r_fail >= r_min)
        assert(r_succ <= r_max and r_succ >= r_min)
        assert(not np.isinf(r_fail))
        assert(not np.isinf(r_succ))

        if (discount == 0):
            val_fail = 0
            val_succ = 0
        else:
            val_fail = r_fail / (1.0 - discount)
            val_succ = r_succ / (1.0 - discount)

        return val_fail, val_succ

    def _build_saver(self):
        vars = self._get_saver_vars()
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_saver_vars(self):
        with self.sess.as_default(), self.graph.as_default():
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            vars = [v for v in vars if '/' + "solvers" + '/' not in v.name]
            # vars = [v for v in vars if '/target/' not in v.name]
            assert len(vars) > 0
        return vars

    def _weight_decay_loss(self, scope):
        vars = self._tf_vars(scope)
        vars_no_bias = [v for v in vars if 'bias' not in v.name]
        loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
        return loss

    def _tf_vars(self, scope=''):
        with self.sess.as_default(), self.graph.as_default():
            res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + scope)
            assert len(res) > 0
        return res

    def end_episode(self):
        if (self.path.pathlength() > 0):
            self._end_path()

            if (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END):
                if (self.enable_training and self.path.pathlength() > 0):
                    self._store_path(self.path)
            elif (self._mode == self.Mode.TEST):
                self._update_test_return(self.path)
            else:
                assert False, Logger.print("Unsupported RL agent mode" + str(self._mode))

            self._update_mode()
        return

    def _end_path(self):
        # 下面的这些先都注释掉
        # s = self._record_state()
        # g = self._record_goal()
        # r = self._record_reward()

        # self.path.rewards.append(r)
        # self.path.states.append(s)
        # self.path.goals.append(g)
        self.path.terminate = self.env.check_terminate(self.id)

    def _store_path(self, path):
        path_id = self.replay_buffer.store(path)
        valid_path = path_id != -1

        if valid_path:
            self.train_return = path.calc_return()

            if self._need_normalizer_update:
                self._record_normalizers(path)

        return path_id

    def _build_bounds(self):
        # self.a_bound_min = self.env.build_action_bound_min(self.id)
        self.a_bound_min = self.build_action_bound_min()
        # self.a_bound_max = self.env.build_action_bound_max(self.id)
        self.a_bound_max = self.build_action_bound_max()
        return

    def _is_first_step(self):
        return len(self.path.states) == 0

    def _update_exp_params(self):
        # anneal_samples是一个常数，数据里面给定的是六千四百万
        lerp = float(self._total_sample_count) / self.exp_anneal_samples
        lerp = np.clip(lerp, 0.0, 1.0)
        self.exp_params_curr = self.exp_params_beg.lerp(self.exp_params_end, lerp)
        return

    def _update_test_return(self, path):
        path_reward = path.calc_return()
        self.test_return += path_reward
        self.test_episode_count += 1
        return

    def _update_mode(self):
        if (self._mode == self.Mode.TRAIN):
            self._update_mode_train()
        elif (self._mode == self.Mode.TRAIN_END):
            self._update_mode_train_end()
        elif (self._mode == self.Mode.TEST):
            self._update_mode_test()
        else:
            assert False, Logger.print("Unsupported RL agent mode" + str(self._mode))
        return

    def _update_mode_train(self):
        return

    def _update_mode_train_end(self):
        self._init_mode_test()
        return

    def _update_mode_test(self):
        if (self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes):
            global_return = MPIUtil.reduce_sum(self.test_return)
            global_count = MPIUtil.reduce_sum(self.test_episode_count)
            avg_return = global_return / global_count
            self.avg_test_return = avg_return

            if self.enable_training:
                self._init_mode_train()
        return

    def _init_mode_train(self):
        self._mode = self.Mode.TRAIN
        self.env.set_mode(self._mode)
        return

    def _init_mode_train_end(self):
        self._mode = self.Mode.TRAIN_END
        return

    def _init_mode_test(self):
        self._mode = self.Mode.TEST
        self.test_return = 0.0
        self.test_episode_count = 0
        # ?为什么env需要mode参数？
        self.env.set_mode(self._mode)
        return

    def _enable_output(self):
        return MPIUtil.is_root_proc() and self.output_dir != ""

    def _enable_int_output(self):
        return MPIUtil.is_root_proc() and self.int_output_dir != ""

    def _update_iter(self, iter):
        if (self._enable_output() and self.iter % self.output_iters == 0):
            output_path = self._get_output_path()
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.save_model(output_path)

        if (self._enable_int_output() and self.iter % self.int_output_iters == 0):
            int_output_path = self._get_int_output_path()
            int_output_dir = os.path.dirname(int_output_path)
            if not os.path.exists(int_output_dir):
                os.makedirs(int_output_dir)
            self.save_model(int_output_path)

        self.iter = iter
        return

    def _enable_draw(self):
        return self.env.enable_draw


class PhyHumanoid(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, show_config=dict(), phy_config=dict()):

        phy_config.setdefault(
            "agent_files", ["data/agents/ct_agent_humanoid_ppo.txt"])
        phy_config.setdefault("char_ctrl_files", [
            "data/controllers/humanoid3d_ctrl.txt"])
        phy_config.setdefault("char_ctrls", ["ct_pd"])
        phy_config.setdefault("char_types", ["general"])
        phy_config.setdefault("character_files", [
            "data/characters/humanoid3d.txt"])
        phy_config.setdefault("enable_char_soft_contact", ["false"])
        phy_config.setdefault("fall_contact_bodies", [
            "0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "12", "13", "14"])
        phy_config.setdefault(
            "motion_file", ["data/motions/humanoid3d_walk.txt"])
        phy_config.setdefault("num_sim_substeps", ["2"])
        phy_config.setdefault("num_update_substeps", ["10"])
        phy_config.setdefault("scene", ["imitate"])
        phy_config.setdefault("sync_char_root_pos", ["true"])
        phy_config.setdefault("sync_char_root_rot", ["false"])
        phy_config.setdefault("terrain_file", ["data/terrain/plane.txt"])
        phy_config.setdefault("train_agents", ["false"])
        phy_config.setdefault("world_scale", ["4"])

        self.enable_draw = show_config.get("enable_draw", False)
        self.show_kin = show_config.get("show_kin", False)
        self.draw_frame_skip = show_config.get("draw_frame_skip", 5)

        # self.enable_draw = config["enable_draw"]
        # self.show_kin = config["show_kin"]

        if self.enable_draw:
            self.num_draw_frame = 0
            self.win_width = 800
            self.win_height = int(self.win_width * 9.0 / 16.0)
            self._init_draw()

        self.playback_speed = 1

        self._core = DeepMimicCore.cDeepMimicCore(self.enable_draw)

        rand_seed = np.random.randint(np.iinfo(np.int32).max)
        self._core.SeedRand(rand_seed)
        # d = Path(__file__).resolve().parents[0]
        # fd = os.open(str(d), os.O_RDONLY)
        # os.fchdir(fd)
        # self._core.ParseArgs(args)

        self._core.SetArgs(phy_config)

        # print("3333333333333333333333")
        # self._core.PrintArgs()
        # print("4444444444444444444444")
        # self.arg_parser = build_arg_parser(args)
        # os.close(fd)
        self._core.Init()
        self._core.SetPlaybackSpeed(self.playback_speed)

        fps = 60
        self.update_timestep = 1.0 / fps

        num_substeps = self._core.GetNumUpdateSubsteps()
        self.timestep = self.update_timestep / num_substeps

        self.agent_id = 0
        self.act_size = self._core.GetActionSize(self.agent_id)
        self.action_space = Box(-2 * np.pi, 2 * np.pi,
                                shape=(self.act_size,), dtype=np.float32)

        self.state_size = self._core.GetStateSize(self.agent_id)
        self.goal_size = self._core.GetGoalSize(self.agent_id)
        self.ground_size = 0
        self.obs_size = self.state_size + self.goal_size + self.ground_size
        self.observation_space = Box(-np.inf, np.inf,
                                     shape=(self.obs_size,), dtype=np.float32)

        if self.enable_draw:
            self.reshaping = False
            self._setup_draw()
            if self.show_kin:
                self._core.Keyboard(107, 0, 0)

    def _init_draw(self):
        glutInit()

        # glutInitContextVersion(3, 2)
        glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
        glutInitContextProfile(GLUT_CORE_PROFILE)

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.win_width, self.win_height)
        glutCreateWindow(b'ExtDeepMimic')
        return

    def _setup_draw(self):
        self._reshape(self.win_width, self.win_height)
        self._core.Reshape(self.win_width, self.win_height)

        return

    def _draw(self):

        if self.num_draw_frame % self.draw_frame_skip == 0:
            self._update_intermediate_buffer()
            self._core.Draw()
            glutSwapBuffers()
            self.reshaping = False
        self.num_draw_frame += 1
        return

    def _reshape(self, w, h):

        self.reshaping = True
        self.win_width = w
        self.win_height = h

        return

    def _shutdown(self):

        print('Shutting down...')
        self._core.Shutdown()
        sys.exit(0)
        return

    def _update_intermediate_buffer(self):
        if not (self.reshaping):
            if (self.win_width != self._core.GetWinWidth() or self.win_height != self._core.GetWinHeight()):
                self._core.Reshape(self.win_width, self.win_height)

        return

    def reset(self):
        self._core.Reset()
        if self.enable_draw:
            self.num_draw_frame = 0
            self._draw()
        obs = self._get_observation()
        return obs

    def step(self, action):
        prev_obs = self._get_observation()
        self._core.SetAction(self.agent_id, np.asarray(action).tolist())

        # self._core.Update(self.timestep)
        time_elapsed = 0.0
        need_update = True
        while need_update:
            self._core.Update(self.timestep)
            time_elapsed += self.timestep
            if self.enable_draw:
                self._draw()

            valid_episode = self._core.CheckValidEpisode()
            if valid_episode:
                done = self._core.IsEpisodeEnd()
                if done:

                    obs = self._get_observation()
                    # 为什么每次都要check observation，有啥用啊
                    valid_obs, obs = self._check_observation(obs, prev_obs)
                    if valid_obs:
                        reward = self._core.CalcReward(self.agent_id)
                        return obs, reward, True, {"valid": 1.0, "time_elapsed": time_elapsed}
                    else:
                        return obs, 0.0, True, {"valid": 0.0, "time_elapsed": time_elapsed}

                else:
                    need_update = not self._core.NeedNewAction(self.agent_id)

            else:

                obs = self._get_observation()
                valid_obs, obs = self._check_observation(obs, prev_obs)
                return obs, 0.0, True, {"valid": 0.0, "time_elapsed": time_elapsed}

        obs = self._get_observation()
        valid_obs, obs = self._check_observation(obs, prev_obs)
        if valid_obs:
            reward = self._core.CalcReward(self.agent_id)
            return obs, reward, False, {"valid": 1.0, "time_elapsed": time_elapsed}
        else:
            return obs, 0.0, True, {"valid": 0.0, "time_elapsed": time_elapsed}

    def _get_observation(self):
        state = np.array(self._core.RecordState(self.agent_id))
        goal = np.array(self._core.RecordGoal(self.agent_id))
        # ground = np.array(self._core.RecordGround(self.agent_id))
        obs = np.concatenate([state, goal], axis=0)
        return obs

    def _check_observation(self, obs, prev_obs):
        if np.isnan(obs).any():
            return False, prev_obs
        else:
            if self.observation_space.contains(obs):
                return True, obs
            else:
                return False, prev_obs

    def get_state_size(self, agent_id):
        return self._core.GetStateSize(agent_id)

    def get_goal_size(self, agent_id):
        return self._core.GetGoalSize(agent_id)

    def get_action_size(self, agent_id):
        return self._core.GetActionSize(agent_id)

    def get_num_actions(self, agent_id):
        return self._core.GetNumActions(agent_id)

    def build_state_offset(self, agent_id):
        # print("state_offset", np.array(self._core.BuildStateOffset(agent_id)))
        return np.array(self._core.BuildStateOffset(agent_id))

    def build_state_scale(self, agent_id):
        # print("state_scale", np.array(self._core.BuildStateScale(agent_id)))
        return np.array(self._core.BuildStateScale(agent_id))

    def build_goal_offset(self, agent_id):
        # print("goal scale", np.array(self._core.BuildGoalOffset(agent_id)))
        return np.array(self._core.BuildGoalOffset(agent_id))

    def build_goal_scale(self, agent_id):
        # print("goal offset", np.array(self._core.BuildGoalScale(agent_id)))
        return np.array(self._core.BuildGoalScale(agent_id))

    def build_action_offset(self, agent_id):
        return np.array(self._core.BuildActionOffset(agent_id))

    def build_action_scale(self, agent_id):
        return np.array(self._core.BuildActionScale(agent_id))

    def build_action_bound_min(self, agent_id):
        return np.array(self._core.BuildActionBoundMin(agent_id))

    def build_action_bound_max(self, agent_id):
        return np.array(self._core.BuildActionBoundMax(agent_id))

    def build_state_norm_groups(self, agent_id):
        # print("state norm groups", np.array(self._core.BuildStateNormGroups(agent_id)))
        return np.array(self._core.BuildStateNormGroups(agent_id))

    def build_goal_norm_groups(self, agent_id):
        # print("goal norm groups", np.array(self._core.BuildGoalNormGroups(agent_id)))
        return np.array(self._core.BuildGoalNormGroups(agent_id))

    def get_reward_min(self, agent_id):
        # print('reward_min', self._core.GetRewardMin(agent_id))
        return self._core.GetRewardMin(agent_id)

    def get_reward_max(self, agent_id):
        # print('reward_max', self._core.GetRewardMax(agent_id))
        return self._core.GetRewardMax(agent_id)

    def set_mode(self, mode):
        self._core.SetMode(mode.value)
        return

    def get_action_space(self, agent_id):
        return ActionSpace(self._core.GetActionSpace(agent_id))

    # 不知道下面这两个到底的作用是什么
    # def get_reward_fail(self, agent_id):
    #     return self._core.GetRewardFail(agent_id)

    # def get_reward_succ(self, agent_id):
    #     return self._core.GetRewardSucc(agent_id)

    def check_terminate(self, agent_id):
        # 没太清楚这个的作用
        return Terminate(self._core.CheckTerminate(agent_id))

    def set_sample_count(self, count):
        self._core.SetSampleCount(count)


# anim
# fps = 60
# update_timestep = 1.0 / fps
# playback_speed = 1
# args = []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('--train', dest='train', action='store_true')

    args = parser.parse_args()

    ppo_config = {
        "ActorNet": "fc_2layers_1024units",
        "ActorStepsize": 0.0000025,
        "ActorMomentum": 0.9,
        "ActorWeightDecay": 0.0005,
        "ActorInitOutputScale": 0.01,

        "CriticNet": "fc_2layers_1024units",
        "CriticStepsize": 0.01,
        "CriticMomentum": 0.9,
        "CriticWeightDecay": 0,

        "UpdatePeriod": 1,
        "ItersPerUpdate": 1,
        "Discount": 0.95,
        "BatchSize": 4096,
        "MiniBatchSize": 256,
        "Epochs": 1,
        "ReplayBufferSize": 500000,
        "InitSamples": 1,
        "NormalizerSamples": 1000000,

        "RatioClip": 0.2,
        "NormAdvClip": 4,
        "TDLambda": 0.95,

        "OutputIters": 10,
        "IntOutputIters": 400,
        "TestEpisodes": 32,

        "ExpAnnealSamples": 64000000,

        "ExpParamsBeg":
        {
            "Rate": 1,
            "InitActionRate": 1,
            "Noise": 0.05,
            "NoiseInternal": 0,
            "Temp": 0.1
        },

        "ExpParamsEnd":
        {
            "Rate": 0.2,
            "InitActionRate": 0.01,
            "Noise": 0.05,
            "NoiseInternal": 0,
            "Temp": 0.001
        }
    }
    if args.train:

        env = PhyHumanoid(phy_config={
            "motion_file": ["data/motions/humanoid3d_walk.txt"],
            "train_agents": ["true"],
            "time_end_lim_exp": ["50"],
            "time_end_lim_max": ["20"],
            "time_end_lim_min": ["20"],
            "time_lim_exp": ["0.2"],
            "time_lim_max": ["0.5"],
            "time_lim_min": ["0.5"],
            "anneal_samples": ["32000000"]})
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        output_path = "output"
        int_output_path = ""

        agent = PPOAgent(env, 0, ppo_config)
        agent.output_dir = output_path
        agent.int_output_path = int_output_path
        agent.enable_training = True
        assert (agent is not None)

        while True:
            agent.collect_experience()
            agent.train()

        return
    else:
        env = PhyHumanoid(phy_config={"motion_file": ["data/motions/humanoid3d_walk.txt"]}, show_config={"enable_draw": True, "show_kin": True, "draw_frame_skip": 5})
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        agent = PPOAgent(env, 0, ppo_config)
        # agent.load_model("data/dm/demo/humanoid3d_walk.ckpt")
        agent.load_model("output/agent0_model.ckpt")
        agent.enable_training = False
        assert (agent is not None)
        for i in range(10000):
            state = env.reset()
            T = 0
            done = False
            while not done:
                action, _ = agent._decide_action(s=state, g=np.zeros((0,), dtype=np.float64))
                # action_eval = action_eval.reshape(env.action_space.shape)
                state, rwd, done, info = env.step(action)
                # print((state[0], rwd, done, info))
                print(T)
                T += 1
            print("==========")


if __name__ == '__main__':
    main()
