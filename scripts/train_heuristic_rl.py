# scripts/train_heuristic_rl.py
import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sixnimmt_env import SixQuiPrendEnv


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


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden_dims(hidden_dims_str: str):
    return [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]


# ============================================================
# Observation wrapper
# Dict obs -> flat Box obs
# Supports:
#   - continuous_obs
#   - sort_rows
#   - add_diff
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
        hand = np.asarray(obs["player_hand"], dtype=np.float32)               # (10,)
        tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32)       # (4,)
        lengths = np.asarray(obs["length_of_rows"], dtype=np.float32)         # (4,)
        bulls = np.asarray(obs["table_bulls"], dtype=np.float32)              # (4,)

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
        assert out.shape[0] == self.obs_dim
        return out


def mask_fn(env):
    # unwrap through Monitor / wrappers
    return np.array(env.unwrapped.action_masks(), dtype=bool)


def make_wrapped_env(
    seed: int,
    continuous_obs: bool,
    sort_rows: bool,
    add_diff: bool,
    monitor_path: str = None,
):
    env = SixQuiPrendEnv()
    env = FlatFeatureObsWrapper(
        env,
        continuous_obs=continuous_obs,
        sort_rows=sort_rows,
        add_diff=add_diff,
    )
    env = Monitor(env, filename=monitor_path)
    env = ActionMasker(env, mask_fn)
    env.reset(seed=seed)
    return env


# ============================================================
# Heuristic policy
# ============================================================

def choose_min_bulls_row_deterministic(base_env):
    """
    Deterministic tie-break for forced eat:
    1) minimum row bulls
    2) minimum row length
    3) minimum row tail
    """
    candidates = []
    for i in range(4):
        bulls = base_env.table.row_bulls[i]
        length = base_env.table.row_lengths[i]
        tail = base_env.table.rows[i][-1].value
        candidates.append((bulls, length, tail, i))
    candidates.sort()
    return candidates[0][-1]


def heuristic_action(base_env):
    """
    Deterministic heuristic:
    For each valid card in agent hand, compute a heuristic cost and choose argmin.

    Intuition:
    - avoid forced eat
    - avoid immediate large penalty
    - prefer small positive gap to row tail
    - avoid long / expensive rows
    - slightly prefer dumping higher-bull cards when safely possible
    """
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

            # forced eat is generally bad
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

            # if row already has 5 cards, adding triggers eating them
            immediate_penalty = (row_bulls[row] / 30.0) if row_lengths[row] >= 5 else 0.0

            cost = (
                3.0 * immediate_penalty
                + 0.80 * gap
                + 0.30 * len_term
                + 0.10 * rowbull_term
                - 0.15 * handbull_term
            )

            # small positive gap is often safe
            raw_gap = card.value - tail
            if 1 <= raw_gap <= 3:
                cost -= 0.15

            # adding the 5th card makes the row dangerous next turn
            if row_lengths[row] == 4:
                cost += 0.10

        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_action = action_idx

    assert best_action is not None
    return best_action


# ============================================================
# Evaluation
# ============================================================

def evaluate_heuristic(
    n_episodes: int,
    seed_start: int,
    continuous_obs: bool,
    sort_rows: bool,
    add_diff: bool,
):
    env = make_wrapped_env(
        seed=seed_start,
        continuous_obs=continuous_obs,
        sort_rows=sort_rows,
        add_diff=add_diff,
        monitor_path=None,
    )
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_return = 0.0

        while not done:
            action = heuristic_action(env.unwrapped)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
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


def evaluate_maskable_ppo(model, env, n_episodes: int, seed_start: int):
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_return = 0.0

        while not done:
            action_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_return += reward
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
# Dataset collection for BC
# ============================================================

def collect_bc_dataset(
    dataset_episodes: int,
    seed_start: int,
    continuous_obs: bool,
    sort_rows: bool,
    add_diff: bool,
):
    env = make_wrapped_env(
        seed=seed_start,
        continuous_obs=continuous_obs,
        sort_rows=sort_rows,
        add_diff=add_diff,
        monitor_path=None,
    )

    obs_list = []
    action_list = []
    mask_list = []
    returns = []

    for ep in range(dataset_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_return = 0.0

        while not done:
            mask = np.array(env.action_masks(), dtype=bool)
            action = heuristic_action(env.unwrapped)

            obs_list.append(obs.copy())
            action_list.append(int(action))
            mask_list.append(mask.copy())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    obs_array = np.stack(obs_list).astype(np.float32)
    actions_array = np.asarray(action_list, dtype=np.int64)
    masks_array = np.stack(mask_list).astype(np.bool_)

    returns = np.asarray(returns, dtype=np.float32)
    dataset_summary = {
        "dataset_episodes": dataset_episodes,
        "num_samples": int(len(obs_array)),
        "heuristic_mean_return": float(returns.mean()),
        "heuristic_std_return": float(returns.std()),
        "heuristic_min_return": float(returns.min()),
        "heuristic_max_return": float(returns.max()),
    }

    return obs_array, actions_array, masks_array, dataset_summary


# ============================================================
# Behavior Cloning pretrain on MaskablePPO policy
# ============================================================

def pretrain_with_bc(
    model,
    obs_array: np.ndarray,
    actions_array: np.ndarray,
    masks_array: np.ndarray,
    bc_epochs: int,
    bc_batch_size: int,
    bc_lr: float,
    run_dir: str,
):
    device = model.device
    model.policy.set_training_mode(True)

    optimizer = optim.Adam(model.policy.parameters(), lr=bc_lr)
    num_samples = len(obs_array)
    indices = np.arange(num_samples)

    bc_log_csv = os.path.join(run_dir, "bc_train_log.csv")

    for epoch in range(1, bc_epochs + 1):
        np.random.shuffle(indices)

        epoch_losses = []
        epoch_accs = []

        for start in range(0, num_samples, bc_batch_size):
            idx = indices[start:start + bc_batch_size]

            batch_obs = torch.tensor(obs_array[idx], dtype=torch.float32, device=device)
            batch_actions = torch.tensor(actions_array[idx], dtype=torch.long, device=device)
            batch_masks = torch.tensor(masks_array[idx], dtype=torch.bool, device=device)

            dist = model.policy.get_distribution(batch_obs, action_masks=batch_masks)
            log_prob = dist.log_prob(batch_actions)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # categorical distribution inside
                pred_actions = dist.distribution.probs.argmax(dim=1)
                acc = (pred_actions == batch_actions).float().mean().item()

            epoch_losses.append(float(loss.item()))
            epoch_accs.append(float(acc))

        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "action_acc": float(np.mean(epoch_accs)),
        }
        append_csv(bc_log_csv, row)

        print(
            f"[BC] epoch={epoch} "
            f"loss={row['loss']:.6f} "
            f"action_acc={row['action_acc']:.4f}"
        )


# ============================================================
# Callback for RL fine-tuning logs + periodic evaluation
# ============================================================

class TrainEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        eval_episodes: int,
        eval_seed_start: int,
        run_dir: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed_start = eval_seed_start
        self.run_dir = run_dir

        self.train_log_csv = os.path.join(run_dir, "rl_train_log.csv")
        self.eval_log_csv = os.path.join(run_dir, "rl_eval_log.csv")

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
            stats = evaluate_maskable_ppo(
                model=self.model,
                env=self.eval_env,
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
                    f"[RL Eval] step={self.num_timesteps} "
                    f"mean_return={stats['mean_return']:.4f} "
                    f"std={stats['std_return']:.4f}"
                )

            if self.best_mean_return is None or stats["mean_return"] > self.best_mean_return:
                self.best_mean_return = stats["mean_return"]
                self.model.save(os.path.join(self.run_dir, "best_finetuned_model"))

        return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/heuristic_rl_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # obs tricks
    parser.add_argument("--continuous_obs", dest="continuous_obs", action="store_true")
    parser.add_argument("--no-continuous_obs", dest="continuous_obs", action="store_false")
    parser.set_defaults(continuous_obs=True)
    parser.add_argument("--sort_rows", action="store_true")
    parser.add_argument("--add_diff", action="store_true")

    # heuristic dataset
    parser.add_argument("--dataset_episodes", type=int, default=5000)
    parser.add_argument("--dataset_seed_start", type=int, default=1000)

    # BC pretrain
    parser.add_argument("--bc_epochs", type=int, default=20)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-3)

    # PPO fine-tune
    parser.add_argument("--ppo_timesteps", type=int, default=100000)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--hidden_dims", type=str, default="256,256")

    # evaluation
    parser.add_argument("--eval_every_steps", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--eval_seed_start", type=int, default=20000)

    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hidden_dims = parse_hidden_dims(args.hidden_dims)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"heuristic_rl_{timestamp}"

    run_dir = os.path.join(args.save_dir, args.run_name)
    ensure_dir(run_dir)

    # save config
    config = vars(args).copy()
    config["device"] = device
    config["hidden_dims"] = hidden_dims
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ------------------------
    # 0) evaluate heuristic alone
    # ------------------------
    heuristic_eval = evaluate_heuristic(
        n_episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
        continuous_obs=args.continuous_obs,
        sort_rows=args.sort_rows,
        add_diff=args.add_diff,
    )
    with open(os.path.join(run_dir, "heuristic_eval.json"), "w", encoding="utf-8") as f:
        json.dump(heuristic_eval, f, indent=2, ensure_ascii=False)

    print("[Heuristic Eval]")
    print(json.dumps(heuristic_eval, indent=2, ensure_ascii=False))

    # ------------------------
    # 1) collect BC dataset
    # ------------------------
    obs_array, actions_array, masks_array, dataset_summary = collect_bc_dataset(
        dataset_episodes=args.dataset_episodes,
        seed_start=args.dataset_seed_start,
        continuous_obs=args.continuous_obs,
        sort_rows=args.sort_rows,
        add_diff=args.add_diff,
    )
    with open(os.path.join(run_dir, "heuristic_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    print("[Dataset Summary]")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    # ------------------------
    # 2) build PPO model
    # ------------------------
    train_monitor_path = os.path.join(run_dir, "train_monitor.csv")

    def train_env_fn():
        return make_wrapped_env(
            seed=args.seed,
            continuous_obs=args.continuous_obs,
            sort_rows=args.sort_rows,
            add_diff=args.add_diff,
            monitor_path=train_monitor_path,
        )

    train_vec_env = DummyVecEnv([train_env_fn])

    eval_env = make_wrapped_env(
        seed=args.eval_seed_start,
        continuous_obs=args.continuous_obs,
        sort_rows=args.sort_rows,
        add_diff=args.add_diff,
        monitor_path=None,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=hidden_dims, vf=hidden_dims)
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=train_vec_env,
        learning_rate=args.ppo_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    # ------------------------
    # 3) BC pretrain
    # ------------------------
    pretrain_with_bc(
        model=model,
        obs_array=obs_array,
        actions_array=actions_array,
        masks_array=masks_array,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        run_dir=run_dir,
    )

    model.save(os.path.join(run_dir, "bc_pretrained_model"))

    # evaluate BC-only before RL
    bc_eval = evaluate_maskable_ppo(
        model=model,
        env=eval_env,
        n_episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
    )
    with open(os.path.join(run_dir, "bc_eval.json"), "w", encoding="utf-8") as f:
        json.dump(bc_eval, f, indent=2, ensure_ascii=False)

    print("[BC Eval]")
    print(json.dumps(bc_eval, indent=2, ensure_ascii=False))

    # ------------------------
    # 4) RL fine-tune
    # ------------------------
    callback = TrainEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        run_dir=run_dir,
        verbose=1,
    )

    model.learn(total_timesteps=args.ppo_timesteps, callback=callback)
    model.save(os.path.join(run_dir, "last_finetuned_model"))

    # final eval for last model
    final_eval = evaluate_maskable_ppo(
        model=model,
        env=eval_env,
        n_episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
    )

    final_summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "heuristic_eval": heuristic_eval,
        "bc_eval": bc_eval,
        "final_eval": final_eval,
        "dataset_summary": dataset_summary,
        "continuous_obs": args.continuous_obs,
        "sort_rows": args.sort_rows,
        "add_diff": args.add_diff,
        "ppo_timesteps": args.ppo_timesteps,
        "bc_epochs": args.bc_epochs,
        "hidden_dims": hidden_dims,
    }

    with open(os.path.join(run_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\n[Final Summary]")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()