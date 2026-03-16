# scripts/eval_policy.py
import argparse
import csv
import json
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sixnimmt_env import SixQuiPrendEnv
from sixnimmt_env.continuous import (
    ContinuousObservationWrapper,
    ContinuousObservationWrapper_addobservation,
    ContinuousObservationWrapper_sorted,
    ContinuousObservationWrapper_sorted_addobservation,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_seed_start", type=int, default=30000)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    wrapper_name = config["wrapper"]

    model_path = os.path.join(args.run_dir, args.checkpoint)
    model = MaskablePPO.load(model_path)

    env = SixQuiPrendEnv()
    env = apply_obs_wrapper(env, wrapper_name)
    env = ActionMasker(env, mask_fn)

    returns = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.eval_seed_start + ep)
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
    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "wrapper": wrapper_name,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_penalty": float(-returns.mean()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(os.path.join(args.run_dir, "external_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.run_dir, "external_eval_episode_returns.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return"])
        writer.writeheader()
        for i, r in enumerate(returns):
            writer.writerow({"episode": i, "return": float(r)})


if __name__ == "__main__":
    main()