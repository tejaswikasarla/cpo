import sys

sys.path.append(".")

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import lasagne.nonlinearities as NL
import numpy as np

# Policy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.gaussian_mlp_baseline import GaussianMLPBaseline

# Environment
from sandbox.cpo.point_gather_env import PointGatherEnv

# Policy optimization
from sandbox.cpo.cpo import CPO
from sandbox.cpo.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.cpo.gather import GatherSafetyConstraint

def run_task(*_):
        trpo_stepsize = 0.01
        trpo_subsample_factor = 0.2
        
        env = PointGatherEnv(apple_reward=10,bomb_cost=1,n_apples=2, activity_range=6)

        policy = GaussianMLPPolicy(env.spec,
                    hidden_sizes=(64,32)
                 )

        baseline = GaussianMLPBaseline (
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )

        safety_baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes': (64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':trpo_stepsize,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    },
            target_key='safety_returns',
            )

        safety_constraint = GatherSafetyConstraint(max_value=0.1, baseline=safety_baseline)

        algo = CPO(
            env=env,
            policy=policy,
            baseline=baseline,
            safety_constraint=safety_constraint,
            safety_gae_lambda=1,
            batch_size=50000,
            max_path_length=15,
            n_itr=100,
            gae_lambda=0.95,
            discount=0.995,
            step_size=trpo_stepsize,
            optimizer_args={'subsample_factor':trpo_subsample_factor},
            #plot=True,
        )

        algo.train()


run_experiment_lite (
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='CPO-PointGather',
    seed=1,
    mode = "local"
    #plot=True
)

