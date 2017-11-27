# Constrained Policy Optimization

Constrained Policy Optimization (CPO) is an algorithm for learning policies that should satisfy behavioral constraints throughout training. [1]

After setting up the required RLLIB and Mujoco modules, clone this repo into <rllib-dir>/sandbox/cpo. 

Then proceed to run cpo in the Point-Gather environment with
```bash
python sandbox/cpo/CPO_point_gather.py 
```

To generate visualization of learned policies, modify the following lines in CPO_point_gather.py file. 
```bash
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
            plot=True,
        )


run_experiment_lite (
    run_task,
    n_parallel=4,
    snapshot_mode="last",
    exp_prefix='CPO-PointGather',
    seed=1,
    mode = "local"
    plot=True
)
```


***

1. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "[Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)". _Proceedings of the 34th International Conference on Machine Learning (ICML), 2017._ 
***

for TiML course project.
