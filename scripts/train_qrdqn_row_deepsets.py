# scripts/train_qrdqn_row_deepsets.py
import argparse
import csv
import json
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sixnimmt_env import SixQuiPrendEnv


# ============================================================
# Utilities
# ============================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_csv(path: str, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_hidden_dims(hidden_dims_str: str):
    return [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]


def get_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ============================================================
# Observation adapter
# 不排序，不加 diff，只做原始 row features + hand features
# 输出一个 flat vector，模型内部再 split
# ============================================================

class RowDeepSetsObservationAdapter:
    """
    Flat layout:
      [ hand_values(10),
        hand_valid_mask(10),
        row_features(4*3) ]
    where each row feature is [tail, length, bulls], normalized.
    Total dim = 10 + 10 + 12 = 32
    """
    def __init__(self):
        self.hand_dim = 10
        self.hand_valid_dim = 10
        self.row_dim = 4 * 3
        self.obs_dim = self.hand_dim + self.hand_valid_dim + self.row_dim

    def preprocess(self, obs: dict) -> np.ndarray:
        hand = np.asarray(obs["player_hand"], dtype=np.float32)  # (10,)
        hand_valid = (hand > 0).astype(np.float32)

        tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32) / 104.0
        lengths = np.asarray(obs["length_of_rows"], dtype=np.float32) / 5.0
        bulls = np.asarray(obs["table_bulls"], dtype=np.float32) / 30.0

        hand_vals = hand / 104.0

        row_feats = np.stack([tails, lengths, bulls], axis=1).reshape(-1).astype(np.float32)

        out = np.concatenate([
            hand_vals.astype(np.float32),
            hand_valid.astype(np.float32),
            row_feats
        ]).astype(np.float32)

        assert out.shape[0] == self.obs_dim
        return out


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, next_mask):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.next_masks[self.ptr] = next_mask

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions[idx], dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device),
            "next_masks": torch.tensor(self.next_masks[idx], dtype=torch.bool, device=device),
        }


# ============================================================
# Row DeepSets QRDQN Network
# ============================================================

def build_mlp(input_dim: int, hidden_dims, output_dim: int):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class RowDeepSetsQRDQNNet(nn.Module):
    """
    Input flat vector:
      hand_vals:     [B, 10]
      hand_valid:    [B, 10]
      row_feats:     [B, 12] -> reshape [B, 4, 3]

    Row encoder:
      phi_row shared on each row
      pool with mean + max
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        head_hidden_dims,
        num_quantiles: int,
        row_embed_dim: int = 64,
        row_phi_hidden_dims=(64, 64),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.row_embed_dim = row_embed_dim

        # shared row encoder phi_row: 3 -> row_embed_dim
        self.phi_row = build_mlp(
            input_dim=3,
            hidden_dims=list(row_phi_hidden_dims),
            output_dim=row_embed_dim,
        )

        # final head input:
        # hand_vals(10) + hand_valid(10) + row_mean(row_embed_dim) + row_max(row_embed_dim)
        head_input_dim = 10 + 10 + row_embed_dim + row_embed_dim

        self.head = build_mlp(
            input_dim=head_input_dim,
            hidden_dims=head_hidden_dims,
            output_dim=action_dim * num_quantiles,
        )

    def forward(self, x):
        """
        x: [B, 32]
        returns: [B, A, N]
        """
        hand_vals = x[:, :10]               # [B, 10]
        hand_valid = x[:, 10:20]            # [B, 10]
        row_feats_flat = x[:, 20:32]        # [B, 12]
        row_feats = row_feats_flat.reshape(-1, 4, 3)  # [B, 4, 3]

        # shared row encoder
        row_emb = self.phi_row(row_feats.reshape(-1, 3))         # [B*4, D]
        row_emb = row_emb.view(-1, 4, self.row_embed_dim)     # [B, 4, D]

        row_mean = row_emb.mean(dim=1)                        # [B, D]
        row_max = row_emb.max(dim=1).values                   # [B, D]

        head_input = torch.cat([hand_vals, hand_valid, row_mean, row_max], dim=1)
        out = self.head(head_input)                           # [B, A*N]
        return out.reshape(-1, self.action_dim, self.num_quantiles)


# ============================================================
# Action selection
# ============================================================

@torch.no_grad()
def select_action(model, obs_vec: np.ndarray, action_mask: np.ndarray, epsilon: float, device: torch.device):
    valid_actions = np.where(action_mask)[0]
    assert len(valid_actions) > 0, "No valid actions available."

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = model(obs_t).mean(dim=-1).squeeze(0)  # [A]
    q_values = q_values.clone()
    q_values[~torch.tensor(action_mask, dtype=torch.bool, device=device)] = -1e9
    return int(torch.argmax(q_values).item())


# ============================================================
# QRDQN loss
# ============================================================

def quantile_huber_loss(pred_quantiles, target_quantiles, kappa: float = 1.0):
    td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # [B, N, N]
    abs_td = td_errors.abs()

    huber = torch.where(
        abs_td <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (abs_td - 0.5 * kappa)
    )

    num_quantiles = pred_quantiles.shape[1]
    taus = (torch.arange(num_quantiles, device=pred_quantiles.device, dtype=pred_quantiles.dtype) + 0.5) / num_quantiles
    taus = taus.view(1, num_quantiles, 1)

    loss = torch.abs(taus - (td_errors.detach() < 0).float()) * huber / kappa
    return loss.mean()


def qrdqn_train_step(online_net, target_net, optimizer, batch, gamma: float, kappa: float = 1.0):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    next_masks = batch["next_masks"]

    all_quantiles = online_net(obs)  # [B, A, N]
    pred_quantiles = all_quantiles[torch.arange(obs.shape[0]), actions]  # [B, N]

    with torch.no_grad():
        next_online_quantiles = online_net(next_obs)          # [B, A, N]
        next_online_q = next_online_quantiles.mean(dim=-1)    # [B, A]
        next_online_q = next_online_q.masked_fill(~next_masks, -1e9)
        next_actions = torch.argmax(next_online_q, dim=1)     # [B]

        next_target_quantiles = target_net(next_obs)[torch.arange(obs.shape[0]), next_actions]  # [B, N]
        target_quantiles = rewards.unsqueeze(1) + gamma * (1.0 - dones.unsqueeze(1)) * next_target_quantiles

    loss = quantile_huber_loss(pred_quantiles, target_quantiles, kappa=kappa)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, adapter, device: torch.device, eval_episodes: int, eval_seed: int):
    env = SixQuiPrendEnv()
    returns = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_vec = adapter.preprocess(obs)
            mask = np.array(env.action_masks(), dtype=bool)
            action = select_action(
                model=model,
                obs_vec=obs_vec,
                action_mask=mask,
                epsilon=0.0,
                device=device,
            )
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/qrdqn_row_deepsets_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=1000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=str, default="512,256")

    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--exploration_fraction", type=float, default=0.3)

    parser.add_argument("--num_quantiles", type=int, default=101)
    parser.add_argument("--kappa", type=float, default=1.0)

    # Row DeepSets specific
    parser.add_argument("--row_embed_dim", type=int, default=64)
    parser.add_argument("--row_phi_hidden_dims", type=str, default="64,64")

    parser.add_argument("--eval_every_steps", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=20000)
    parser.add_argument("--log_every_episodes", type=int, default=20)

    args = parser.parse_args()

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    row_phi_hidden_dims = parse_hidden_dims(args.row_phi_hidden_dims)
    device = get_device(args.device)
    set_global_seed(args.seed)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"qrdqn_rowdeepsets_{timestamp}"

    run_dir = os.path.join(args.save_dir, args.run_name)
    ensure_dir(run_dir)

    config = vars(args).copy()
    config["hidden_dims"] = hidden_dims
    config["row_phi_hidden_dims"] = row_phi_hidden_dims
    config["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    adapter = RowDeepSetsObservationAdapter()
    obs_dim = adapter.obs_dim
    action_dim = 10

    env = SixQuiPrendEnv()

    online_net = RowDeepSetsQRDQNNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        head_hidden_dims=hidden_dims,
        num_quantiles=args.num_quantiles,
        row_embed_dim=args.row_embed_dim,
        row_phi_hidden_dims=row_phi_hidden_dims,
    ).to(device)

    target_net = RowDeepSetsQRDQNNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        head_hidden_dims=hidden_dims,
        num_quantiles=args.num_quantiles,
        row_embed_dim=args.row_embed_dim,
        row_phi_hidden_dims=row_phi_hidden_dims,
    ).to(device)

    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)

    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim)

    train_log_csv = os.path.join(run_dir, "train_log.csv")
    eval_log_csv = os.path.join(run_dir, "eval_log.csv")

    init_eval = evaluate(
        model=online_net,
        adapter=adapter,
        device=device,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
    )
    append_csv(eval_log_csv, {"global_step": 0, "episode": 0, **init_eval})
    print(f"[Init Eval] mean_return={init_eval['mean_return']:.4f}, std={init_eval['std_return']:.4f}")

    best_mean_return = init_eval["mean_return"]
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": online_net.state_dict(),
            "config": config,
        },
        best_ckpt_path,
    )

    global_step = 0
    episode_idx = 0
    recent_returns = deque(maxlen=20)
    last_loss = None
    start_time = time.time()
    exploration_steps = max(1, int(args.exploration_fraction * args.total_steps))

    while global_step < args.total_steps:
        obs, _ = env.reset(seed=args.seed + episode_idx)
        obs_vec = adapter.preprocess(obs)
        done = False

        ep_return = 0.0
        ep_len = 0

        while not done and global_step < args.total_steps:
            mask = np.array(env.action_masks(), dtype=bool)

            if global_step < exploration_steps:
                ratio = global_step / exploration_steps
                epsilon = args.eps_start + ratio * (args.eps_end - args.eps_start)
            else:
                epsilon = args.eps_end

            action = select_action(
                model=online_net,
                obs_vec=obs_vec,
                action_mask=mask,
                epsilon=epsilon,
                device=device,
            )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_obs_vec = adapter.preprocess(next_obs)
            next_mask = np.array(env.action_masks(), dtype=bool)

            replay.add(
                obs=obs_vec,
                action=action,
                reward=reward,
                next_obs=next_obs_vec,
                done=float(done),
                next_mask=next_mask,
            )

            obs_vec = next_obs_vec
            ep_return += reward
            ep_len += 1
            global_step += 1

            if (
                global_step >= args.learning_starts
                and len(replay) >= args.batch_size
                and global_step % args.train_freq == 0
            ):
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size, device)
                    last_loss = qrdqn_train_step(
                        online_net=online_net,
                        target_net=target_net,
                        optimizer=optimizer,
                        batch=batch,
                        gamma=args.gamma,
                        kappa=args.kappa,
                    )

            if global_step % args.target_update_interval == 0:
                target_net.load_state_dict(online_net.state_dict())

            if global_step % args.eval_every_steps == 0:
                eval_stats = evaluate(
                    model=online_net,
                    adapter=adapter,
                    device=device,
                    eval_episodes=args.eval_episodes,
                    eval_seed=args.eval_seed,
                )
                append_csv(eval_log_csv, {
                    "global_step": global_step,
                    "episode": episode_idx,
                    **eval_stats
                })

                print(
                    f"[Eval] step={global_step} "
                    f"mean_return={eval_stats['mean_return']:.4f} "
                    f"std={eval_stats['std_return']:.4f}"
                )

                if eval_stats["mean_return"] > best_mean_return:
                    best_mean_return = eval_stats["mean_return"]
                    torch.save(
                        {
                            "model_state_dict": online_net.state_dict(),
                            "config": config,
                        },
                        best_ckpt_path,
                    )

        recent_returns.append(ep_return)

        append_csv(train_log_csv, {
            "global_step": global_step,
            "episode": episode_idx,
            "episode_len": ep_len,
            "episode_return": float(ep_return),
            "recent20_avg_return": float(np.mean(recent_returns)),
            "epsilon": float(epsilon),
            "loss": "" if last_loss is None else float(last_loss),
        })

        if (episode_idx + 1) % args.log_every_episodes == 0:
            print(
                f"[Train] episode={episode_idx + 1} "
                f"step={global_step} "
                f"ep_return={ep_return:.4f} "
                f"recent20={np.mean(recent_returns):.4f} "
                f"epsilon={epsilon:.4f} "
                f"loss={last_loss}"
            )

        episode_idx += 1

    last_ckpt_path = os.path.join(run_dir, "last_model.pt")
    torch.save(
        {
            "model_state_dict": online_net.state_dict(),
            "config": config,
        },
        last_ckpt_path,
    )

    final_eval = evaluate(
        model=online_net,
        adapter=adapter,
        device=device,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
    )

    final_summary = {
        "run_name": args.run_name,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "best_mean_return": best_mean_return,
        "final_mean_return": final_eval["mean_return"],
        "final_std_return": final_eval["std_return"],
        "elapsed_seconds": time.time() - start_time,
        "lr": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "num_quantiles": args.num_quantiles,
        "hidden_dims": hidden_dims,
        "row_embed_dim": args.row_embed_dim,
        "row_phi_hidden_dims": row_phi_hidden_dims,
    }

    with open(os.path.join(run_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()