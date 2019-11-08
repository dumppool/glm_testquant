from gym.spaces import Box
# import gym
# import numpy as np
# import os
# from extdm import DeepMimicCore
# from pathlib import Path
# from extdm.util.arg_parser import ArgParser
# from extdm.util.logger import Logger
# import extdm.util.mpi_util as MPIUtil
# from ray.rllib.env import MultiAgentEnv
from PhysicsCore import DeepMimicCore
import numpy as np
import os
import gym
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *
# import logging

# # Dimensions of the window we are drawing into.
# win_width = 800
# win_height = int(win_width * 9.0 / 16.0)
# reshaping = False

# # anim
# fps = 60
# update_timestep = 1.0 / fps
# display_anim_time = int(1000 * update_timestep)
# animating = True

# logger = logging.getLogger(__name__)


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


def fc_net(input, layers_sizes, activation, reuse=None, flatten=False):  # build fully connected network
    curr_tf = input
    for i, size in enumerate(layers_sizes):
        with tf.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.layers.dense(inputs=curr_tf,
                                      units=size,
                                      kernel_initializer=xavier_initializer,
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


class Agent(object):
    """docstring for Agent"""

    def __init__(self, arg):
        super(Agent, self).__init__(in_path)
        self.arg = arg

        with self.sess.as_default(), self.graph.as_default():
            self.saver.restore(self.sess, in_path)
            self._load_normalizers()
            Logger.print('Model loaded from: ' + in_path)

    def _build_graph(self, json_data):
        with self.sess.as_default(), self.graph.as_default():
            with tf.variable_scope(self.tf_scope):
                self._build_nets(json_data)
                self._initialize_vars()
                self._build_saver()
        return

    def _initialize_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def _build_nets(self, json_data):

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
        self.g_tf = tf.placeholder(tf.float32, shape=(
            [None, g_size] if self.has_goal() else None), name="g")
        self.old_logp_tf = tf.placeholder(
            tf.float32, shape=[None], name="old_logp")
        self.exp_mask_tf = tf.placeholder(
            tf.float32, shape=[None], name="exp_mask")

        with tf.variable_scope('main'):
            with tf.variable_scope('actor'):
                self.a_mean_tf = self._build_net_actor(actor_init_output_scale)
            with tf.variable_scope('critic'):
                self.critic_tf = self._build_net_critic()

        if self.a_mean_tf is not None:
            Logger.print('Built actor net!')

        if self.critic_tf is not None:
            Logger.print('Built critic net!')

        self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)
        norm_a_noise_tf = self.norm_a_std_tf * \
            tf.random_normal(shape=tf.shape(self.a_mean_tf))
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
        self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(
            x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf)

        return

    def _build_net_actor(self, init_output_scale):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]

        h = build_net(input_tfs)
        norm_a_tf = tf.layers.dense(inputs=h, units=self.get_action_size(), activation=None,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale))

        a_tf = self.a_norm.unnormalize_tf(norm_a_tf)
        return a_tf

    def _build_net_critic(self):
        norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
        input_tfs = [norm_s_tf]
        if (self.has_goal()):
            norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
            input_tfs += [norm_g_tf]

        h = build_net(input_tfs)
        norm_val_tf = tf.layers.dense(inputs=h, units=1, activation=None,
                                      kernel_initializer=TFUtil.xavier_initializer)

        norm_val_tf = tf.reshape(norm_val_tf, [-1])
        val_tf = self.val_norm.unnormalize_tf(norm_val_tf)
        return val_tf

    def _init_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            # update normalizers to sync the tensorflow tensors
            self.s_norm.update()
            self.g_norm.update()
            self.a_norm.update()
        return


if __name__ == '__main__':

    env = PhyHumanoid({"enable_draw": True, "show_kin": True,
                       "draw_frame_skip": 1, "config_file": "run_humanoid3d_output_args.txt"})
    for i in range(10):
        print("============================")
        obs = env.reset()
        done = False
        while not done:
            print(done)
            obs, reward, done, info = env.step(
                np.zeros(env.action_space.shape))
