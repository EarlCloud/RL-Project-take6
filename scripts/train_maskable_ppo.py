# scripts/train_maskable_ppo.py
import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sixnimmt_env import SixQuiPrendEnv
from sixnimmt_env.continuous import (
    ContinuousObservationWrapper,
    ContinuousObservationWrapper_addobservation,
    ContinuousObservationWrapper_sorted,
    ContinuousObservationWrapper_sorted_addobservation,
)


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_csv(path: str, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mask_fn(env):
    return np.array(env.unwrapped.action_masks(), dtype=bool)


def apply_obs_wrapper(env, wrapper_name: str):
    if wrapper_name == "none":
        return env
    elif wrapper_name == "continuous":
        return ContinuousObservationWrapper(env)
    elif wrapper_name == "addobservation":
        return ContinuousObservationWrapper_addobservation(env)
    elif wrapper_name == "sorted":
        return ContinuousObservationWrapper_sorted(env)
    elif wrapper_name == "sorted_addobservation":
        return ContinuousObservationWrapper_sorted_addobservation(env)
    else:
        raise ValueError(f"Unknown wrapper_name={wrapper_name}")


def make_env(seed: int, wrapper_name: str, monitor_path: str = None):
    def _init():
        env = SixQuiPrendEnv()
        env = apply_obs_wrapper(env, wrapper_name)
        env = Monitor(env, filename=monitor_path)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init


def evaluate_model(model, wrapper_name: str, n_episodes: int, seed_start: int):
    env = SixQuiPrendEnv()
    env = apply_obs_wrapper(env, wrapper_name)
    env = ActionMasker(env, mask_fn)

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_return = 0.0

        while not done:
            action_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, r, terminated, truncated, _ = env.step(int(action))
            ep_return += r
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.asarray(returns, dtype=np.float32)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_penalty": float(-returns.mean()),
    }


# ============================================================
# Callback
# ============================================================

class PPOMonitorEvalCallback(BaseCallback):
    def __init__(
        self,
        wrapper_name: str,
        eval_freq: int,
        eval_episodes: int,
        eval_seed_start: int,
        run_dir: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.wrapper_name = wrapper_name
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed_start = eval_seed_start
        self.run_dir = run_dir

        self.train_log_csv = os.path.join(run_dir, "train_log.csv")
        self.eval_log_csv = os.path.join(run_dir, "eval_log.csv")

        self.best_mean_return = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                row = {
                    "timesteps": self.num_timesteps,
                    "episode_reward": float(info["episode"]["r"]),
                    "episode_length": int(info["episode"]["l"]),
                }
                append_csv(self.train_log_csv, row)

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            stats = evaluate_model(
                model=self.model,
                wrapper_name=self.wrapper_name,
                n_episodes=self.eval_episodes,
                seed_start=self.eval_seed_start,
            )
            row = {
                "timesteps": self.num_timesteps,
                **stats,
            }
            append_csv(self.eval_log_csv, row)

            if self.verbose:
                print(
                    f"[PPO Eval] step={self.num_timesteps} "
                    f"mean_return={stats['mean_return']:.4f} "
                    f"std={stats['std_return']:.4f}"
                )

            if self.best_mean_return is None or stats["mean_return"] > self.best_mean_return:
                self.best_mean_return = stats["mean_return"]
                self.model.save(os.path.join(self.run_dir, "best_model"))

        return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/ppo_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrapper",
        type=str,
        default="sorted_addobservation",
        choices=["none", "continuous", "addobservation", "sorted", "sorted_addobservation"],
    )

    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--progress_bar", action="store_true")

    parser.add_argument("--eval_every_steps", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--eval_seed_start", type=int, default=20000)

    args = parser.parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"ppo_{timestamp}"

    run_dir = os.path.join(args.save_dir, args.run_name)
    ensure_dir(run_dir)

    config = vars(args).copy()
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_monitor_path = os.path.join(run_dir, "train_monitor.csv")
    vec_env = DummyVecEnv([
        make_env(seed=args.seed, wrapper_name=args.wrapper, monitor_path=train_monitor_path)
    ])

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
    )

    callback = PPOMonitorEvalCallback(
        wrapper_name=args.wrapper,
        eval_freq=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        run_dir=run_dir,
        verbose=1,
    )

    start_time = time.time()

    # initial eval
    init_eval = evaluate_model(
        model=model,
        wrapper_name=args.wrapper,
        n_episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
    )
    append_csv(os.path.join(run_dir, "eval_log.csv"), {"timesteps": 0, **init_eval})

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=callback,
        progress_bar=args.progress_bar,
    )

    model.save(os.path.join(run_dir, "last_model"))

    final_eval = evaluate_model(
        model=model,
        wrapper_name=args.wrapper,
        n_episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
    )

    final_summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "wrapper": args.wrapper,
        "total_timesteps": int(args.total_timesteps),
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "best_mean_return": callback.best_mean_return,
        "final_mean_return": final_eval["mean_return"],
        "final_std_return": final_eval["std_return"],
        "elapsed_seconds": time.time() - start_time,
    }

    with open(os.path.join(run_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\n[PPO Final Summary]")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()