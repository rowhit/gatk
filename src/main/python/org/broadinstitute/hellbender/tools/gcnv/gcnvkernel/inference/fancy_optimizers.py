from abc import abstractmethod
from collections import OrderedDict
from functools import partial

import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
from pymc3.variational.updates import get_or_compute_grads

from .. import types
from ..io import io_commons
from ..models.fancy_model import GeneralizedContinuousModel


class FancyStochasticOptimizer:
    @abstractmethod
    def get_opt(self,
                model: GeneralizedContinuousModel=None,
                approx: pm.MeanField=None):
        raise NotImplementedError

    @staticmethod
    def get_call_kwargs(_locals_):
        _locals_ = _locals_.copy()
        _locals_.pop('loss_or_grads')
        _locals_.pop('params')
        return _locals_


class FancyAdamax(FancyStochasticOptimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, sample_specific=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.sample_specific = sample_specific
        self.m = []
        self.u = []

    @staticmethod
    def sample_specific_adamax(loss_or_grads=None,
                               params=None,
                               model: GeneralizedContinuousModel=None,
                               approx: pm.MeanField=None,
                               learning_rate=0.002, beta1=0.9,
                               beta2=0.999, epsilon=1e-8,
                               sample_specific=False,
                               base_class: 'FancyAdamax' = None):
        if loss_or_grads is None and params is None:
            return partial(FancyAdamax.sample_specific_adamax,
                           **FancyStochasticOptimizer.get_call_kwargs(locals()))
        elif loss_or_grads is None or params is None:
            raise ValueError(
                'Please provide both `loss_or_grads` and `params` to get updates')
        assert model is not None
        assert approx is not None

        all_grads = get_or_compute_grads(loss_or_grads, params)
        t_prev = th.shared(pm.theanof.floatX(0.))
        updates = OrderedDict()

        # indices of sample-specific vars
        if sample_specific:
            vmap_list = io_commons.get_var_map_list_from_meanfield_approx(approx)
            sample_specific_indices = []
            for vmap in vmap_list:
                if vmap.var in model.sample_specific_var_registry:
                    sample_specific_indices += [idx for idx in range(vmap.slc.start, vmap.slc.stop)]
            update_indices = th.shared(np.asarray(sample_specific_indices, dtype=np.int))
            num_dof = len(sample_specific_indices)

        # Using theano constant to prevent upcasting of float32
        one = tt.constant(1)

        t = t_prev + 1
        a_t = learning_rate / (one - beta1**t)

        for param, g_t in zip(params, all_grads):
            if sample_specific:
                g_t_view = g_t[update_indices]
                m_prev = th.shared(np.zeros((num_dof,), dtype=types.floatX),
                                   broadcastable=(False,))
                u_prev = th.shared(np.zeros((num_dof,), dtype=types.floatX),
                                   broadcastable=(False,))
            else:
                g_t_view = g_t
                value = param.get_value(borrow=True)
                m_prev = th.shared(np.zeros(value.shape, dtype=types.floatX),
                                   broadcastable=(False,))
                u_prev = th.shared(np.zeros(value.shape, dtype=types.floatX),
                                   broadcastable=(False,))

            # save a reference to m and u in the base class
            if base_class is not None:
                base_class.m.append(m_prev)
                base_class.u.append(u_prev)

            m_t = beta1 * m_prev + (one - beta1) * g_t_view
            u_t = tt.maximum(beta2 * u_prev, abs(g_t_view))
            step = a_t * m_t / (u_t + epsilon)

            if sample_specific:
                new_param = tt.inc_subtensor(param[update_indices], -step)
            else:
                new_param = param - step

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = new_param

        updates[t_prev] = t
        return updates

    def get_opt(self,
                model: GeneralizedContinuousModel=None,
                approx: pm.MeanField=None):
        return FancyAdamax.sample_specific_adamax(model=model, approx=approx,
                                                  beta1=self.beta1, beta2=self.beta2,
                                                  learning_rate=self.learning_rate,
                                                  sample_specific=self.sample_specific,
                                                  base_class=self)