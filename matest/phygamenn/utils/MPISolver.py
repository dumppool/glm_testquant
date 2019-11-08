from mpi4py import MPI
import tensorflow as tf
import numpy as np
import phygamenn.utils.MPIUtils as MPIUtil
from phygamenn.utils.Logger import Logger

from abc import ABC, abstractmethod


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int)
               for a in out), "shape function assumes that shape is fully known"
    return out


def intprod(x):
    return int(np.prod(x))


def numel(x):
    n = intprod(var_shape(x))
    return n


def flatten(arr_list):
    return np.concatenate([np.reshape(a, [-1]) for a in arr_list], axis=0)


class SetFromFlat(object):
    def __init__(self, sess, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.sess = sess
        self.theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []

        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(
                self.theta[start:start + size], shape)))
            start += size

        self.op = tf.group(*assigns)

        return

    def __call__(self, theta):
        self.sess.run(self.op, feed_dict={self.theta: theta})
        return


class GetFlat(object):
    def __init__(self, sess, var_list):
        self.sess = sess
        self.op = tf.concat(
            axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        return

    def __call__(self):
        return self.sess.run(self.op)


class Solver(ABC):
    def __init__(self, vars):
        self.vars = vars
        return

    @abstractmethod
    def update(self, grads):
        pass


class MPISolver(Solver):
    CHECK_SYNC_ITERS = 1000

    def __init__(self, sess, optimizer, vars):
        super().__init__(vars)
        self.sess = sess
        self.optimizer = optimizer
        self._build_grad_feed(vars)
        self._update = optimizer.apply_gradients(
            zip(self._grad_tf_list, self.vars))
        self._set_flat_vars = SetFromFlat(sess, self.vars)
        self._get_flat_vars = GetFlat(sess, self.vars)

        self.iter = 0
        grad_dim = self._calc_grad_dim()
        self._flat_grad = np.zeros(grad_dim, dtype=np.float32)
        self._global_flat_grad = np.zeros(grad_dim, dtype=np.float32)
        return

    def get_stepsize(self):
        return self.optimizer._learning_rate_tensor.eval()

    def update(self, grads=None, grad_scale=1.0):
        if grads is not None:
            self._flat_grad = flatten(grads)
        else:
            self._flat_grad.fill(0)
        return self.update_flatgrad(self._flat_grad, grad_scale)

    def update_flatgrad(self, flat_grad, grad_scale=1.0):
        # print("update_flatgrad")
        if self.iter % self.CHECK_SYNC_ITERS == 0:
            # print(self.check_synced())
            assert self.check_synced(), Logger.print('Network parameters desynchronized')
        if grad_scale != 1.0:
            flat_grad *= grad_scale

        MPI.COMM_WORLD.Allreduce(flat_grad, self._global_flat_grad, op=MPI.SUM)
        self._global_flat_grad /= MPIUtil.get_num_procs()

        self._load_flat_grad(self._global_flat_grad)
        self.sess.run([self._update], self._grad_feed)
        self.iter += 1

        return

    def sync(self):
        vars = self._get_flat_vars()
        MPIUtil.bcast(vars)
        self._set_flat_vars(vars)
        return

    def check_synced(self):
        synced = True
        if self._is_root():
            # print("here")
            vars = self._get_flat_vars()
            MPIUtil.bcast(vars)
        else:
            vars_local = self._get_flat_vars()
            vars_root = np.empty_like(vars_local)
            # print("vars_local before", vars_local)
            # print("vars_root before", vars_local)
            MPIUtil.bcast(vars_root)
            # print("vars_root after", vars_root)
            # print("vars_local after", vars_local)
            synced = (vars_local == vars_root).all()
        return synced

    def _is_root(self):
        return MPIUtil.is_root_proc()

    def _build_grad_feed(self, vars):
        self._grad_tf_list = []
        self._grad_buffers = []
        for v in self.vars:
            shape = v.get_shape()
            grad = np.zeros(shape)
            grad_tf = tf.placeholder(tf.float32, shape=shape)
            self._grad_buffers.append(grad)
            self._grad_tf_list.append(grad_tf)

        self._grad_feed = dict({g_tf: g for g_tf, g in zip(
            self._grad_tf_list, self._grad_buffers)})

        return

    def _calc_grad_dim(self):
        grad_dim = 0
        for grad in self._grad_buffers:
            grad_dim += grad.size
        return grad_dim

    def _load_flat_grad(self, flat_grad):
        start = 0
        for g in self._grad_buffers:
            size = g.size
            np.copyto(g, np.reshape(flat_grad[start:start + size], g.shape))
            start += size
        return
