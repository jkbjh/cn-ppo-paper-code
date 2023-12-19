# -----
"""Policies: abstract base class and concrete implementations."""
import argparse

import gym.wrappers
import numpy as np
import stable_baselines3.common.utils
from cnppo.cnpolicy import ColoredNoiseActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

# -----
"""
VERSION 1.1: add SDE support as a baseline
VERSION 1.2: add n_steps configuration support
VERSION 1.3: changed backward version support.
"""
# -----

VERSION = 1.3
SUPPORTED_VERSIONS = {1.1, 1.2, VERSION}


def run_experiment(
    env_id="Pendulum-v1",
    seed=1234,
    n_envs=50,
    n_eval_envs=10,
    clip_range=0.2,
    lr=1e-4,
    ent_coef=0.0,
    gamma=0.99,
    gae_lambda=0.95,
    batch_size=64,
    n_epochs=10,
    noise_color=0.0,
    eval_episodes=50,
    cn_policy=True,
    total_timesteps=1e7,
    eval_freq: int = 10_240,
    n_steps: int = 2048,
    use_sde=False,
):
    print(f"using seed {seed}")
    stable_baselines3.common.utils.set_random_seed(seed)
    # Create the environment
    # FlattenObservation wrapper is necessary because dm-control environments (with dm2gym) return dict-based space.
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed, wrapper_class=gym.wrappers.FlattenObservation)
    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=seed, wrapper_class=gym.wrappers.FlattenObservation)

    policy_kwargs = {}
    if cn_policy:
        policy = ColoredNoiseActorCriticPolicy
        policy_kwargs["noise_color_beta"] = noise_color
        policy_kwargs["noise_rng"] = np.random.default_rng(seed=seed)
    else:
        policy = "MlpPolicy"
    agent = stable_baselines3.ppo.PPO(
        env=env,
        policy=policy,
        policy_kwargs=policy_kwargs,
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_epochs=n_epochs,
        n_steps=n_steps,
        ent_coef=ent_coef,
        learning_rate=lr,
        clip_range=clip_range,
        tensorboard_log=f"tblogs/nc{noise_color}",
        verbose=1,
        use_sde=use_sde,  # --- V1.1
    )
    eval_cb = stable_baselines3.common.callbacks.EvalCallback(
        eval_env=eval_env, n_eval_episodes=eval_episodes, eval_freq=max(eval_freq // n_envs, 1)
    )
    agent.learn(total_timesteps=int(total_timesteps), progress_bar=True, callback=[eval_cb])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--n-envs", type=int, default=50)
    parser.add_argument("--n-eval-envs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.5e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=10_240)
    parser.add_argument("--no-cn-policy", action="store_true")
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--total-timesteps", type=float, default=int(2_048_010))
    parser.add_argument("--noise-color", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=2048)  # 2048 is the stable-baselines default
    # ----------
    parser.add_argument("--version", type=float, default=VERSION)

    args = parser.parse_args()
    if args.use_sde and not args.no_cn_policy:
        parser.error("Use of state-dependent-exploration SDE can only be used with --no-cn-policy.")
    if args.version not in SUPPORTED_VERSIONS:
        raise AssertionError(
            (
                f"ERROR: you requested version {args.version}, "
                f"however the code is version {VERSION}, "
                f"only {SUPPORTED_VERSIONS} are supported."
            )
        )
    print(f"step: 0\nVERSION: {VERSION}")
    run_experiment(
        env_id=args.env_id,
        seed=args.seed,
        n_envs=args.n_envs,
        n_eval_envs=args.n_eval_envs,
        clip_range=args.clip_range,
        lr=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_episodes=args.eval_episodes,
        cn_policy=not args.no_cn_policy,
        noise_color=args.noise_color,
        total_timesteps=int(args.total_timesteps),
        eval_freq=args.eval_freq,
        use_sde=args.use_sde,
        n_steps=args.n_steps,
    )
