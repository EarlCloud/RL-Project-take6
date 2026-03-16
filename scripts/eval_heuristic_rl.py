# scripts/eval_heuristic_rl.py
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sixnimmt_env import SixQuiPrendEnv


# ============================================================
# Utilities
# ============================================================

def append_csv(path: str, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# Same wrapper / mask / heuristic as training
# ============================================================

class FlatFeatureObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, continuous_obs=True, sort_rows=False, add_diff=False):
        super().__init__(env)
        self.continuous_obs = continuous_obs
        self.sort_rows = sort_rows
        self.add_diff = add_diff

        self.base_dim = 10 + 4 + 4 + 4
        self.diff_dim = 10 * 4 if add_diff else 0
        self.obs_dim = self.base_dim + self.diff_dim

        if continuous_obs:
            low = np.full((self.obs_dim,), -1.0, dtype=np.float32)
            high = np.full((self.obs_dim,), 1.0, dtype=np.float32)
        else:
            low = np.full((self.obs_dim,), -104.0, dtype=np.float32)
            high = np.full((self.obs_dim,), 104.0, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        hand = np.asarray(obs["player_hand"], dtype=np.float32)
        tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32)
        lengths = np.asarray(obs["length_of_rows"], dtype=np.float32)
        bulls = np.asarray(obs["table_bulls"], dtype=np.float32)

        if self.sort_rows:
            order = np.argsort(tails)
            tails = tails[order]
            lengths = lengths[order]
            bulls = bulls[order]

        valid_hand = hand > 0

        if self.continuous_obs:
            hand_feat = hand / 104.0
            tails_feat = tails / 104.0
            lengths_feat = lengths / 5.0
            bulls_feat = bulls / 30.0
        else:
            hand_feat = hand
            tails_feat = tails
            lengths_feat = lengths
            bulls_feat = bulls

        feats = [hand_feat, tails_feat, lengths_feat, bulls_feat]

        if self.add_diff:
            diff = np.zeros((10, 4), dtype=np.float32)
            for i in range(10):
                if valid_hand[i]:
                    diff[i] = hand[i] - tails
            if self.continuous_obs:
                diff = diff / 104.0
            feats.append(diff.reshape(-1))

        out = np.concatenate(feats).astype(np.float32)
        return out


def mask_fn(env):
    return np.array(env.unwrapped.action_masks(), dtype=bool)


def make_wrapped_env(seed, continuous_obs, sort_rows, add_diff):
    env = SixQuiPrendEnv()
    env = FlatFeatureObsWrapper(
        env,
        continuous_obs=continuous_obs,
        sort_rows=sort_rows,
        add_diff=add_diff,
    )
    env = ActionMasker(env, mask_fn)
    env.reset(seed=seed)
    return env


def choose_min_bulls_row_deterministic(base_env):
    candidates = []
    for i in range(4):
        bulls = base_env.table.row_bulls[i]
        length = base_env.table.row_lengths[i]
        tail = base_env.table.rows[i][-1].value
        candidates.append((bulls, length, tail, i))
    candidates.sort()
    return candidates[0][-1]


def heuristic_action(base_env):
    hand = base_env.players[0].hand
    row_bulls = base_env.table.row_bulls
    row_lengths = base_env.table.row_lengths
    rows = base_env.table.rows

    best_action = None
    best_cost = None

    for action_idx, card in enumerate(hand):
        forced_row = base_env.table.get_forced_row(card)

        if forced_row == -1:
            row = choose_min_bulls_row_deterministic(base_env)
            immediate_penalty = row_bulls[row] / 30.0
            len_term = row_lengths[row] / 5.0
            rowbull_term = row_bulls[row] / 30.0
            handbull_term = card.bulls / 7.0
            cost = (
                3.0 * immediate_penalty
                + 0.4 * len_term
                + 0.2 * rowbull_term
                - 0.10 * handbull_term
                + 0.50
            )
        else:
            row = forced_row
            tail = rows[row][-1].value
            gap = (card.value - tail) / 104.0
            len_term = row_lengths[row] / 5.0
            rowbull_term = row_bulls[row] / 30.0
            handbull_term = card.bulls / 7.0
            immediate_penalty = (row_bulls[row] / 30.0) if row_lengths[row] >= 5 else 0.0

            cost = (
                3.0 * immediate_penalty
                + 0.80 * gap
                + 0.30 * len_term
                + 0.10 * rowbull_term
                - 0.15 * handbull_term
            )

            raw_gap = card.value - tail
            if 1 <= raw_gap <= 3:
                cost -= 0.15

            if row_lengths[row] == 4:
                cost += 0.10

        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_action = action_idx

    return best_action


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["heuristic", "ppo"], required=True)
    parser.add_argument("--checkpoint", type=str, default="best_finetuned_model")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_seed_start", type=int, default=30000)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    env = make_wrapped_env(
        seed=args.eval_seed_start,
        continuous_obs=config["continuous_obs"],
        sort_rows=config["sort_rows"],
        add_diff=config["add_diff"],
    )

    returns = []

    model = None
    if args.mode == "ppo":
        model_path = os.path.join(args.run_dir, args.checkpoint)
        model = MaskablePPO.load(model_path)

    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.eval_seed_start + ep)
        done = False
        ep_return = 0.0

        while not done:
            if args.mode == "heuristic":
                action = heuristic_action(env.unwrapped)
            else:
                action_mask = env.action_masks()
                action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
                action = int(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.asarray(returns, dtype=np.float32)
    summary = {
        "run_dir": args.run_dir,
        "mode": args.mode,
        "checkpoint": args.checkpoint if args.mode == "ppo" else None,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_penalty": float(-returns.mean()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = os.path.join(args.run_dir, f"external_eval_{args.mode}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.run_dir, f"external_eval_episode_returns_{args.mode}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return"])
        writer.writeheader()
        for i, r in enumerate(returns):
            writer.writerow({"episode": i, "return": float(r)})


if __name__ == "__main__":
    main()