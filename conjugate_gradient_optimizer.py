from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.misc.ext import sliced_fun
from _ast import Num

class PerlmutterHvp(Serializable):

    def __init__(self, num_slices=1):
        Serializable.quick_init(self, locals())
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params(trainable=True)

        constraint_grads = theano.grad(
            f, wrt=params, disconnected_inputs='warn')
        xs = tuple([ext.new_tensor_like("%s x" % p.name, p) for p in params])

        def Hx_plain():
            Hx_plain_splits = TT.grad(
                TT.sum([TT.sum(g * x)
                        for g, x in zip(constraint_grads, xs)]),
                wrt=params,
                disconnected_inputs='warn'
            )
            return TT.concatenate([TT.flatten(s) for s in Hx_plain_splits])

        self.opt_fun = ext.lazydict(
            f_Hx_plain=lambda: ext.compile_function(
                inputs=inputs + xs,
                outputs=Hx_plain(),
                log_name="f_Hx_plain",
            ),
        )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(
                inputs, xs) + self.reg_coeff * x
            return ret

        return eval
        
class ConjugateGradientOptimizer(Serializable):
    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1):
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def update_opt(self, loss, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint", *args,
                   **kwargs):

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        constraint_term, constraint_value = leq_constraint

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params, disconnected_inputs='warn')
        flat_grad = ext.flatten_tensor_variables(grads)

        self._hvp_approach.update_opt(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_constraint"], self._num_slices)(inputs, extra_inputs)

    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("computing loss before")
        loss_before = sliced_fun(self._opt_fun["f_loss"], self._num_slices)(
            inputs, extra_inputs)
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = sliced_fun(self._opt_fun["f_grad"], self._num_slices)(
            inputs, extra_inputs)

        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)

        approx_g = Hx(descent_direction)
        q = descent_direction.dot(approx_g)
        residual = np.sqrt((approx_g - flat_g).dot(approx_g - flat_g))
        rescale  = q / (descent_direction.dot(descent_direction))
        logger.record_tabular("OptimDiagnostic_Residual",residual)
        logger.record_tabular("OptimDiagnostic_Rescale", rescale)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        prev_param = np.copy(self._target.get_param_values(trainable=True))
        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = sliced_fun(
                self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
                self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" %
                           self._constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._constraint_name)
            self._target.set_param_values(prev_param, trainable=True)