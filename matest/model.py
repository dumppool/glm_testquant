import tensorflow as tf

import os
import pickle
import numpy as np

import phygamenn.utils.MpiUtils as MPIUtil
from phygamenn.utils.Logger import Logger

from gym.spaces import Box
from PhysicsCore import DeepMimicCore
import gym
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


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


def build_net(input_tfs, reuse=False):
    layers = [1024, 512]
    activation = tf.nn.relu

    input_tf = tf.concat(axis=-1, values=input_tfs)
    h = fc_net(input_tf, layers, activation=activation, reuse=reuse)
    h = activation(h)
    return h


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


class Agent(object):
    RESOURCE_SCOPE = 'resource'
    SOLVER_SCOPE = 'solvers'
    # def __init__(self, env, ckpt_location):

    def __init__(self, ckpt_location=""):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.tf_scope = 'agent'
        self.discount = 0.95
        self.id = 0
        self._exp_action = False
        self.ckpt_location = ckpt_location
        # self.env = env
        self._build_normalizers()
        self._build_bounds()

        self._build_graph()
        self._build_saver()
        if ckpt_location is not '':
            self.restore_model(self.ckpt_location)

    def _build_graph(self):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_nets()

    def _build_normalizers(self):
        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
            with tf.variable_scope(self.RESOURCE_SCOPE):
                val_offset, val_scale = self._calc_val_offset_scale(
                    self.discount)
                self.val_norm = TFNormalizer(self.sess, 'val_norm', 1)
                self.val_norm.set_mean_std(-val_offset, 1.0 / val_scale)

                get_state_size = 197
                build_state_norm_groups = np.zeros((197,))
                build_state_norm_groups[0] = -1
                # self.s_norm = TFNormalizer(self.sess, 's_norm', self.env.get_state_size(self.id),
                #                            self.env.build_state_norm_groups(self.id))
                self.s_norm = TFNormalizer(
                    self.sess, 's_norm', get_state_size, build_state_norm_groups)

                build_state_offset = np.zeros((197,))
                build_state_offset[0] = -0.5
                build_state_scale = np.ones((197,))
                build_state_scale[0] = 2
                # self.s_norm.set_mean_std(-self.env.build_state_offset(self.id),
                #                          1 / self.env.build_state_scale(self.id))
                self.s_norm.set_mean_std(-build_state_offset,
                                         1 / build_state_scale)

                get_goal_size = 0
                build_goal_norm_groups = np.array(())
                # self.g_norm = TFNormalizer(self.sess, 'g_norm', self.env.get_goal_size(self.id),
                #                            self.env.build_goal_norm_groups(self.id))
                self.g_norm = TFNormalizer(
                    self.sess, 'g_norm', get_goal_size, build_goal_norm_groups)

                build_goal_offset = np.array(())
                build_goal_scale = np.array(())
                # self.g_norm.set_mean_std(-self.env.build_goal_offset(self.id),
                #                          1 / self.env.build_goal_scale(self.id))
                self.g_norm.set_mean_std(-build_goal_offset,
                                         1 / build_goal_scale)

                get_action_size = 36
                # self.a_norm = TFNormalizer(self.sess, 'a_norm', self.env.get_action_size(self.id))
                self.a_norm = TFNormalizer(
                    self.sess, 'a_norm', get_action_size)

                # self.a_norm.set_mean_std(-self.env.build_action_offset(self.id),
                #                          1 / self.env.build_action_scale(self.id))
                self.a_norm.set_mean_std(-self.build_action_offset(),
                                         1 / self.build_action_scale())

    def build_action_offset(self):
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

    def build_action_scale(self):
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

    def _build_nets(self):

        actor_init_output_scale = 0.01

        s_size = 197
        g_size = 0
        a_size = 36

        # setup input tensors
        self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
        self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self.tar_val_tf = tf.placeholder(
            tf.float32, shape=[None], name="tar_val")
        self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
        self.g_tf = tf.placeholder(tf.float32, shape=(None), name="g")
        self.old_logp_tf = tf.placeholder(
            tf.float32, shape=[None], name="old_logp")
        self.exp_mask_tf = tf.placeholder(
            tf.float32, shape=[None], name="exp_mask")

        with tf.variable_scope('main'):
            with tf.variable_scope('actor'):
                self.a_mean_tf = self._build_net_actor(actor_init_output_scale)
            with tf.variable_scope('critic'):
                self.critic_tf = self._build_net_critic()

        self.norm_a_std_tf = 0.05 * tf.ones(a_size)
        norm_a_noise_tf = self.norm_a_std_tf * \
            tf.random_normal(shape=tf.shape(self.a_mean_tf))
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
        self.sample_a_logp_tf = calc_logp_gaussian(
            x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf)

        return

    def _build_net_actor(self, init_output_scale):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]

        h = build_net(input_tfs, False)
        norm_a_tf = tf.layers.dense(inputs=h, units=self.get_action_size(), activation=None,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale,
                                                                                     maxval=init_output_scale))

        a_tf = self.a_norm.unnormalize_tf(norm_a_tf)
        return a_tf

    def _build_net_critic(self):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]

        h = build_net(input_tfs, False)
        norm_val_tf = tf.layers.dense(inputs=h, units=1, activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())

        norm_val_tf = tf.reshape(norm_val_tf, [-1])
        val_tf = self.val_norm.unnormalize_tf(norm_val_tf)
        return val_tf

    def get_state_size(self):
        return 197

    def get_goal_size(self):
        return 0

    def get_action_size(self):
        return 36

    def actor_predict(self, status):
        s = np.reshape(status, [-1, self.get_state_size()])
        g = None
        feed = {
            self.s_tf: s,
            self.g_tf: g,
            self.exp_mask_tf: np.array([0])
        }
        a, logp = self.sess.run(
            [self.sample_a_tf, self.sample_a_logp_tf], feed_dict=feed)
        action = a
        return action

    def restore_model(self, ckpt_location):
        with self.sess.as_default(), self.graph.as_default():
            print(self.graph)
            self.saver.restore(self.sess, ckpt_location)
            self._load_normalizers()
        return

    def _load_normalizers(self):
        self.s_norm.load()
        self.g_norm.load()
        self.a_norm.load()

    def _calc_val_offset_scale(self, discount):
        val_min, val_max = self._calc_val_bounds(discount)
        val_offset = 0
        val_scale = 1

        if (np.isfinite(val_min) and np.isfinite(val_max)):
            val_offset = -0.5 * (val_max + val_min)
            val_scale = 2 / (val_max - val_min)

        return val_offset, val_scale

    def _calc_val_bounds(self, discount):
        # r_min = self.env.get_reward_min(self.id)
        r_min = 0.0
        # r_max = self.env.get_reward_max(self.id)
        r_max = 1.0
        assert (r_min <= r_max)

        val_min = r_min / (1.0 - discount)
        val_max = r_max / (1.0 - discount)
        return val_min, val_max

    def _build_bounds(self):
        # self.a_bound_min = self.env.build_action_bound_min(self.id)
        self.a_bound_min = self.build_action_bound_min()
        # self.a_bound_max = self.env.build_action_bound_max(self.id)
        self.a_bound_max = self.build_action_bound_max()
        return

    def build_action_bound_min(self):
        res = np.ones((36,))
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

    def _build_saver(self):
        vars = self._get_variables()
        self.saver = tf.train.Saver(vars, max_to_keep=0)
        return

    def _get_variables(self):
        with self.sess.as_default(), self.graph.as_default():
            # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = [v for v in vars if '/' +
                    self.SOLVER_SCOPE + '/' not in v.name]
            assert len(vars) > 0
        return vars

    def _get_values_of_variables(self):
        variable = []
        with self.sess.as_default(), self.graph.as_default():
            # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars = [v for v in vars if '/' +
                    self.SOLVER_SCOPE + '/' not in v.name]
            for i in range(len(vars)):
                bil = self.sess.run(vars[i])
                variable.append(bil)
            assert len(vars) > 0
        return variable

    def _set_variable_test(self):
        vars = self._get_variables()
        for i in range(len(vars)):
            self.sess.run(tf.assign(vars[i], np.ones(shape=vars[i].shape)))
        pass

    def _set_variable_from_pkl(self, filename):

        with open(filename, "rb") as file:
            values = pickle.load(file)
        vars = self._get_variables()
        for i in range(len(vars)):
            self.sess.run(tf.assign(vars[i], values[i]))

        pass

    def _set_variable_from_buffer(self, values):
        vars = self._get_variables()
        for i in range(len(vars)):
            self.sess.run(tf.assign(vars[i], values[i]))

    @property
    def values(self):
        return self._get_values_of_variables()

    # def ckpt2pkl(self, in_path):
        # txtpath = "/home/liang/extdm/extdm/data/policies/pkl"
        # filename = os.path.splitext(os.path.basename(in_path))[0]
        # filename = os.path.join(txtpath, filename) + '.pkl'

        # self.restore_model(in_path)
        # values = self._get_values_of_variables()
        # print(values)
        # with open(out_path, 'wb') as file:
        #     pickle.dump(values, file)

        # pass


class PhyHumanoid(gym.Env):

    def __init__(self, config):

        args = ['--arg_file']
        args.append(os.path.join("data", config["config_file"]))

        self.enable_draw = config.get("enable_draw", False)
        self.show_kin = config.get("show_kin", False)
        self.draw_frame_skip = config.get("draw_frame_skip", 5)

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

        self._core.ParseArgs(args)

        self._core.Init()
        self._core.SetPlaybackSpeed(self.playback_speed)

        fps = 60
        self.update_timestep = 1.0 / fps

        num_substeps = self._core.GetNumUpdateSubsteps()
        self.timestep = self.update_timestep / num_substeps

        self.agent_id = 0
        self.act_size = self._core.GetActionSize(self.agent_id)
        print(self.act_size)
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
        need_update = True
        while need_update:
            self._core.Update(self.timestep)
            if self.enable_draw:
                self._draw()

            valid_episode = self._core.CheckValidEpisode()
            if valid_episode:
                done = self._core.IsEpisodeEnd()
                if done:

                    obs = self._get_observation()
                    valid_obs, obs = self._check_observation(obs, prev_obs)
                    if valid_obs:
                        reward = self._core.CalcReward(self.agent_id)
                        return obs, reward, True, {"valid": 1.0}
                    else:
                        return obs, 0.0, True, {"valid": 0.0}

                else:
                    need_update = not self._core.NeedNewAction(self.agent_id)

            else:

                obs = self._get_observation()
                valid_obs, obs = self._check_observation(obs, prev_obs)
                return obs, 0.0, True, {"valid": 0.0}

        obs = self._get_observation()
        valid_obs, obs = self._check_observation(obs, prev_obs)
        if valid_obs:
            reward = self._core.CalcReward(self.agent_id)
            return obs, reward, False, {"valid": 1.0}
        else:
            return obs, 0.0, True, {"valid": 0.0}

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
        return np.array(self._core.BuildStateOffset(agent_id))

    def build_state_scale(self, agent_id):
        return np.array(self._core.BuildStateScale(agent_id))

    def build_goal_offset(self, agent_id):
        return np.array(self._core.BuildGoalOffset(agent_id))

    def build_goal_scale(self, agent_id):
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
        return np.array(self._core.BuildStateNormGroups(agent_id))

    def build_goal_norm_groups(self, agent_id):
        return np.array(self._core.BuildGoalNormGroups(agent_id))

    def get_reward_min(self, agent_id):
        return self._core.GetRewardMin(agent_id)

    def get_reward_max(self, agent_id):
        return self._core.GetRewardMax(agent_id)


if __name__ == '__main__':
    agent = Agent(
        "data/dm/demo/humanoid3d_spinkick.ckpt")

    env = PhyHumanoid({
        "enable_draw": True,
        "show_kin": True,
        "draw_frame_skip": 1,
        "config_file": "run_humanoid3d_output_args.txt"})

    for i in range(10):
        obs = env.reset()
        T = 0
        done = False
        while not done:
            action = agent.actor_predict(obs)
            action = action.reshape(env.action_space.shape)
            obs, rwd, done, info = env.step(action)
            print((obs[0], rwd, done, info))
            print(T)
            T += 1
        print("==========")
