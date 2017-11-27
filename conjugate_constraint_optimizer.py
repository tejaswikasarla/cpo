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

class ConjugateConstraintOptimizer(Serializable):
    def __init__(
            self,
            cg_iters=10,
            verbose_cg=False,
            resample_inputs=False,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1,
            linesearch_infeasible_recovery=True):
    
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._verbose_cg = verbose_cg
        self._resample_inputs = resample_inputs
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach


    def update_opt(self, loss, target, quad_leq_constraint, lin_leq_constraint, inputs, 
                    extra_inputs=None, 
                    constraint_name_1="quad_constraint",
                    constraint_name_2="lin_constraint", 
                    using_surrogate=False,
                    true_linear_leq_constraint=None,
                    precompute=False,
                    attempt_feasible_recovery=False,
                    attempt_infeasible_recovery=False,
                    revert_to_last_safe_point=False,
                    *args, **kwargs):

        self.precompute = precompute
        self.attempt_feasible_recovery = attempt_feasible_recovery
        self.attempt_infeasible_recovery = attempt_infeasible_recovery
        self.revert_to_last_safe_point = revert_to_last_safe_point

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        constraint_term_1, constraint_value_1 = quad_leq_constraint
        constraint_term_2, constraint_value_2 = lin_leq_constraint

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params, disconnected_inputs='warn')
        flat_grad = ext.flatten_tensor_variables(grads)

        lin_constraint_grads = theano.grad(constraint_term_2, wrt=params, disconnected_inputs='warn')
        flat_lin_constraint_grad = ext.flatten_tensor_variables(lin_constraint_grads)

        if using_surrogate and not(precompute):
            constraint_term_2 = true_linear_leq_constraint

        self._hvp_approach.update_opt(f=constraint_term_1, target=target, 
                                      inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_quad_constraint_val = constraint_value_1
        self._max_lin_constraint_val = constraint_value_2
        self._constraint_name_1 = constraint_name_1
        self._constraint_name_2 = constraint_name_2

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
            f_quad_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term_1,
                log_name="quad_constraint",
            ),
            f_lin_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term_2,
                log_name="lin_constraint",
            ),
            f_lin_constraint_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_lin_constraint_grad,
                log_name="lin_constraint_grad",
            ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term_1, constraint_term_2],
                log_name="f_loss_constraint",
            ),
        )

        self.last_safe_point = None
        self._last_lin_pred_S = 0
        self._last_surr_pred_S = 0

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

    def optimize(self, 
                 inputs, 
                 extra_inputs=None, 
                 subsample_grouped_inputs=None, 
                 precomputed_eval=None, 
                 precomputed_threshold=None,
                 diff_threshold=False,
                 inputs2=None,
                 extra_inputs2=None,
                ):

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if inputs2 is None:
            inputs2 = inputs
        if extra_inputs2 is None:
            extra_inputs2 = tuple()

        def subsampled_inputs(inputs,subsample_grouped_inputs):
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
            return subsample_inputs

        subsample_inputs = subsampled_inputs(inputs,subsample_grouped_inputs)
        if self._resample_inputs:
            subsample_inputs2 = subsampled_inputs(inputs,subsample_grouped_inputs)

        loss_before = sliced_fun(self._opt_fun["f_loss"], self._num_slices)(
            inputs, extra_inputs)

        flat_g = sliced_fun(self._opt_fun["f_grad"], self._num_slices)(
            inputs, extra_inputs)
        flat_b = sliced_fun(self._opt_fun["f_lin_constraint_grad"], self._num_slices)(
            inputs2, extra_inputs2)

        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
        v = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters, verbose=self._verbose_cg)

        approx_g = Hx(v)
        q = v.dot(approx_g) 
        delta = 2 * self._max_quad_constraint_val
 
        eps = 1e-8

        residual = np.sqrt((approx_g - flat_g).dot(approx_g - flat_g))
        rescale  = q / (v.dot(v))

        if self.precompute:
            S = precomputed_eval
            assert(np.ndim(S)==0) 
        else:
            S = sliced_fun(self._opt_fun["lin_constraint"], self._num_slices)(inputs, extra_inputs) 

        c = S - self._max_lin_constraint_val
        if c > 0:
            logger.log("warning! safety constraint is already violated")
        else:
            self.last_safe_point = np.copy(self._target.get_param_values(trainable=True))

        stop_flag = False

        if flat_b.dot(flat_b) <= eps :
            lam = np.sqrt(q / delta)
            nu = 0
            w = 0
            r,s,A,B = 0,0,0,0
            optim_case = 4
        else:
            if self._resample_inputs:
                Hx = self._hvp_approach.build_eval(subsample_inputs2 + extra_inputs)

            norm_b = np.sqrt(flat_b.dot(flat_b))
            unit_b = flat_b / norm_b
            w = norm_b * krylov.cg(Hx, unit_b, cg_iters=self._cg_iters, verbose=self._verbose_cg)

            r = w.dot(approx_g) # approx = b^T H^{-1} g
            s = w.dot(Hx(w))    # approx = b^T H^{-1} b

            A = q - r**2 / s                # this should always be positive by Cauchy-Schwarz
            B = delta - c**2 / s            # this one says whether or not the closest point on the plane is feasible

            if c <0 and B < 0:
                optim_case = 3
            elif c < 0 and B > 0:
                optim_case = 2
            elif c > 0 and B > 0:
                optim_case = 1
                if self.attempt_feasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting feasible recovery")
                else:
                    logger.log("alert! problem is feasible but needs recovery, and we were instructed not to attempt recovery")
                    stop_flag = True
            else:
                optim_case = 0
                if self.attempt_infeasible_recovery:
                    logger.log("alert! conjugate constraint optimizer is attempting infeasible recovery")
                else:
                    logger.log("alert! problem is infeasible, and we were instructed not to attempt recovery")
                    stop_flag = True

            lam = np.sqrt(q / delta)
            nu  = 0

            if optim_case == 2 or optim_case == 1:
                lam_mid = r / c
                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                lam_a = np.sqrt(A / (B + eps))
                L_a = -np.sqrt(A*B) - r*c / (s + eps)                 

                lam_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                if lam_mid > 0:
                    if c < 0:
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid
                    else:
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)
        nextS = S + np.sqrt(delta * s)


        def record_zeros():
            logger.record_tabular("BacktrackIters", 0)
            logger.record_tabular("LossRejects", 0)
            logger.record_tabular("QuadRejects", 0)
            logger.record_tabular("LinRejects", 0)


        if optim_case > 0:
            flat_descent_step = (1. / (lam + eps) ) * ( v + nu * w )
        else:
            flat_descent_step = np.sqrt(delta / (s + eps)) * w

        prev_param = np.copy(self._target.get_param_values(trainable=True))

        prev_lin_constraint_val = sliced_fun(
            self._opt_fun["f_lin_constraint"], self._num_slices)(inputs, extra_inputs)

        lin_reject_threshold = self._max_lin_constraint_val
        if precomputed_threshold is not None:
            lin_reject_threshold = precomputed_threshold
        if diff_threshold:
            lin_reject_threshold += prev_lin_constraint_val

        def check_nan():
            loss, quad_constraint_val, lin_constraint_val = sliced_fun(
                self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
            if np.isnan(loss) or np.isnan(quad_constraint_val) or np.isnan(lin_constraint_val):
                if np.isnan(loss):
                    logger.log("Violated because loss is NaN")
                if np.isnan(quad_constraint_val):
                    logger.log("Violated because quad_constraint %s is NaN" %
                               self._constraint_name_1)
                if np.isnan(lin_constraint_val):
                    logger.log("Violated because lin_constraint %s is NaN" %
                               self._constraint_name_2)
                self._target.set_param_values(prev_param, trainable=True)

        def line_search(check_loss=True, check_quad=True, check_lin=True):
            loss_rejects = 0
            quad_rejects = 0
            lin_rejects  = 0
            n_iter = 0
            for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                self._target.set_param_values(cur_param, trainable=True)
                loss, quad_constraint_val, lin_constraint_val = sliced_fun(
                    self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
                loss_flag = loss < loss_before
                quad_flag = quad_constraint_val <= self._max_quad_constraint_val
                lin_flag  = lin_constraint_val  <= lin_reject_threshold
                if check_loss and not(loss_flag):
                    loss_rejects += 1
                if check_quad and not(quad_flag):
                    quad_rejects += 1
                if check_lin and not(lin_flag):
                    lin_rejects += 1

                if (loss_flag or not(check_loss)) and (quad_flag or not(check_quad)) and (lin_flag or not(check_lin)):
                    break

            return loss, quad_constraint_val, lin_constraint_val, n_iter


        def wrap_up():
            if optim_case < 4:
                lin_constraint_val = sliced_fun(
                    self._opt_fun["f_lin_constraint"], self._num_slices)(inputs, extra_inputs)
                lin_constraint_delta = lin_constraint_val - prev_lin_constraint_val
                logger.record_tabular("LinConstraintDelta",lin_constraint_delta)

                cur_param = self._target.get_param_values()
                
                next_linear_S = S + flat_b.dot(cur_param - prev_param)
                next_surrogate_S = S + lin_constraint_delta

                lin_surrogate_acc = 100.*(next_linear_S - next_surrogate_S) / next_surrogate_S

                lin_pred_err = (self._last_lin_pred_S - S) #/ (S + eps)
                surr_pred_err = (self._last_surr_pred_S - S) #/ (S + eps)
                self._last_lin_pred_S = next_linear_S
                self._last_surr_pred_S = next_surrogate_S

            else:
                lin_pred_err = (self._last_lin_pred_S - 0) #/ (S + eps)
                surr_pred_err = (self._last_surr_pred_S - 0) #/ (S + eps)
                self._last_lin_pred_S = 0
                self._last_surr_pred_S = 0

        if stop_flag==True:
            record_zeros()
            wrap_up()
            return

        if optim_case == 1 and not(self.revert_to_last_safe_point):
            if self._linesearch_infeasible_recovery:
                logger.log("feasible recovery mode: constrained natural gradient step. performing linesearch on constraints.")
                line_search(False,True,True)
            else:
                self._target.set_param_values(prev_param - flat_descent_step, trainable=True)
                logger.log("feasible recovery mode: constrained natural gradient step. no linesearch performed.")
            check_nan()
            record_zeros()
            wrap_up()
            return
        elif optim_case == 0 and not(self.revert_to_last_safe_point):
            if self._linesearch_infeasible_recovery:
                logger.log("infeasible recovery mode: natural safety step. performing linesearch on constraints.")
                line_search(False,True,True)
            else:
                self._target.set_param_values(prev_param - flat_descent_step, trainable=True)
                logger.log("infeasible recovery mode: natural safety gradient step. no linesearch performed.")
            check_nan()
            record_zeros()
            wrap_up()
            return
        elif (optim_case == 0 or optim_case == 1) and self.revert_to_last_safe_point:
            if self.last_safe_point:
                self._target.set_param_values(self.last_safe_point, trainable=True)
                logger.log("infeasible recovery mode: reverted to last safe point!")
            else:
                logger.log("alert! infeasible recovery mode failed: no last safe point to revert to.")
            record_zeros()
            wrap_up()
            return


        loss, quad_constraint_val, lin_constraint_val, n_iter = line_search()

        if (np.isnan(loss) or np.isnan(quad_constraint_val) or np.isnan(lin_constraint_val) or loss >= loss_before 
            or quad_constraint_val >= self._max_quad_constraint_val
            or lin_constraint_val > lin_reject_threshold) and not self._accept_violation:
            self._target.set_param_values(prev_param, trainable=True)
        wrap_up()
        
