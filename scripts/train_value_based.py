# scripts/train_value_based.py
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
# 不改环境，只在算法侧把 Dict observation -> flat vector
# 支持 continuous / sort_rows / add_diff
# ============================================================

class ObservationAdapter:
    def __init__(self, continuous_obs=True, sort_rows=False, add_diff=False):
        self.continuous_obs = continuous_obs
        self.sort_rows = sort_rows
        self.add_diff = add_diff

        self.base_dim = 10 + 4 + 4 + 4   # hand + tails + lengths + bulls
        self.diff_dim = 10 * 4 if add_diff else 0
        self.obs_dim = self.base_dim + self.diff_dim

    def preprocess(self, obs: dict) -> np.ndarray:
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

        batch = {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions[idx], dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32, device=device),
            "next_masks": torch.tensor(self.next_masks[idx], dtype=torch.bool, device=device),
        }
        return batch


# ============================================================
# Networks
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


class DQNNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, action_dim)

    def forward(self, x):
        return self.net(x)  # [B, A]


class QRDQNNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims, num_quantiles: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.net = build_mlp(obs_dim, hidden_dims, action_dim * num_quantiles)

    def forward(self, x):
        out = self.net(x)  # [B, A*N]
        return out.view(-1, self.action_dim, self.num_quantiles)  # [B, A, N]


# ============================================================
# Action selection with mask
# ============================================================

def compute_q_values(model, obs_tensor: torch.Tensor, algo: str):
    """
    obs_tensor: [B, obs_dim]
    returns: [B, action_dim]
    """
    if algo == "dqn":
        return model(obs_tensor)
    elif algo == "qrdqn":
        quantiles = model(obs_tensor)          # [B, A, N]
        return quantiles.mean(dim=-1)          # [B, A]
    else:
        raise ValueError(f"Unknown algo: {algo}")


@torch.no_grad()
def select_action(model, obs_vec: np.ndarray, action_mask: np.ndarray, epsilon: float, device: torch.device, algo: str):
    valid_actions = np.where(action_mask)[0]
    assert len(valid_actions) > 0, "No valid actions available."

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
    q_values = compute_q_values(model, obs_t, algo=algo).squeeze(0)                  # [A]
    q_values = q_values.clone()
    q_values[~torch.tensor(action_mask, dtype=torch.bool, device=device)] = -1e9
    return int(torch.argmax(q_values).item())


# ============================================================
# Losses / updates
# ============================================================

def dqn_train_step(online_net, target_net, optimizer, batch, gamma: float):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    next_masks = batch["next_masks"]

    q_all = online_net(obs)                                      # [B, A]
    q = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)         # [B]

    with torch.no_grad():
        next_q_online = online_net(next_obs)                     # [B, A]
        next_q_online = next_q_online.masked_fill(~next_masks, -1e9)
        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)    # [B, 1]

        next_q_target = target_net(next_obs).gather(1, next_actions).squeeze(1)  # [B]
        target = rewards + gamma * (1.0 - dones) * next_q_target

    loss = F.smooth_l1_loss(q, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def quantile_huber_loss(pred_quantiles, target_quantiles, kappa: float = 1.0):
    """
    pred_quantiles:   [B, N]
    target_quantiles: [B, N]
    """
    # delta_{ij} = target_j - pred_i
    td_errors = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)   # [B, N, N]
    abs_td = td_errors.abs()

    huber = torch.where(
        abs_td <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (abs_td - 0.5 * kappa)
    )

    num_quantiles = pred_quantiles.shape[1]
    taus = (torch.arange(num_quantiles, device=pred_quantiles.device, dtype=pred_quantiles.dtype) + 0.5) / num_quantiles
    taus = taus.view(1, num_quantiles, 1)  # [1, N, 1]

    loss = torch.abs(taus - (td_errors.detach() < 0).float()) * huber / kappa
    return loss.mean()


def qrdqn_train_step(online_net, target_net, optimizer, batch, gamma: float, num_quantiles: int, kappa: float = 1.0):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_obs = batch["next_obs"]
    dones = batch["dones"]
    next_masks = batch["next_masks"]

    all_quantiles = online_net(obs)   # [B, A, N]
    pred_quantiles = all_quantiles[torch.arange(obs.shape[0]), actions]  # [B, N]

    with torch.no_grad():
        next_online_quantiles = online_net(next_obs)         # [B, A, N]
        next_online_q = next_online_quantiles.mean(dim=-1)   # [B, A]
        next_online_q = next_online_q.masked_fill(~next_masks, -1e9)
        next_actions = torch.argmax(next_online_q, dim=1)    # [B]

        next_target_quantiles = target_net(next_obs)[torch.arange(obs.shape[0]), next_actions]   # [B, N]
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
def evaluate(model, adapter, algo: str, device: torch.device, eval_episodes: int, eval_seed: int):
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
                algo=algo,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.array(returns, dtype=np.float32)
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

    # --------------- basic ---------------
    parser.add_argument("--algo", type=str, choices=["dqn", "qrdqn"], default="dqn")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/value_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])

    # --------------- observation tricks ---------------
    parser.add_argument("--continuous_obs", dest="continuous_obs", action="store_true")
    parser.add_argument("--no-continuous_obs", dest="continuous_obs", action="store_false")
    parser.set_defaults(continuous_obs=True)

    parser.add_argument("--sort_rows", action="store_true")
    parser.add_argument("--add_diff", action="store_true")

    # --------------- training ---------------
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_starts", type=int, default=2_000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--target_update_interval", type=int, default=1_000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dims", type=str, default="256,256")

    # --------------- exploration ---------------
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--exploration_fraction", type=float, default=0.30)

    # --------------- QRDQN only ---------------
    parser.add_argument("--num_quantiles", type=int, default=51)
    parser.add_argument("--kappa", type=float, default=1.0)

    # --------------- eval / save ---------------
    parser.add_argument("--eval_every_steps", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=10_000)
    parser.add_argument("--log_every_episodes", type=int, default=20)

    args = parser.parse_args()

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    device = get_device(args.device)
    set_global_seed(args.seed)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.algo}_{timestamp}"

    run_dir = os.path.join(args.save_dir, args.run_name)
    ensure_dir(run_dir)

    # Save config
    config = vars(args).copy()
    config["hidden_dims"] = hidden_dims
    config["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Adapter / env
    adapter = ObservationAdapter(
        continuous_obs=args.continuous_obs,
        sort_rows=args.sort_rows,
        add_diff=args.add_diff,
    )
    obs_dim = adapter.obs_dim
    action_dim = 10

    env = SixQuiPrendEnv()

    # Networks
    if args.algo == "dqn":
        online_net = DQNNet(obs_dim, action_dim, hidden_dims).to(device)
        target_net = DQNNet(obs_dim, action_dim, hidden_dims).to(device)
    else:
        online_net = QRDQNNet(obs_dim, action_dim, hidden_dims, args.num_quantiles).to(device)
        target_net = QRDQNNet(obs_dim, action_dim, hidden_dims, args.num_quantiles).to(device)

    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=args.lr)

    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim)

    train_log_csv = os.path.join(run_dir, "train_log.csv")
    eval_log_csv = os.path.join(run_dir, "eval_log.csv")

    # Initial evaluation before training
    init_eval = evaluate(
        model=online_net,
        adapter=adapter,
        algo=args.algo,
        device=device,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
    )
    init_eval_row = {
        "global_step": 0,
        "episode": 0,
        **init_eval,
    }
    append_csv(eval_log_csv, init_eval_row)
    print(f"[Init Eval] mean_return={init_eval['mean_return']:.4f}, std={init_eval['std_return']:.4f}")

    best_mean_return = init_eval["mean_return"]
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": online_net.state_dict(),
            "algo": args.algo,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "num_quantiles": args.num_quantiles,
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
            current_mask = np.array(env.action_masks(), dtype=bool)

            # linear epsilon schedule
            if global_step < exploration_steps:
                ratio = global_step / exploration_steps
                epsilon = args.eps_start + ratio * (args.eps_end - args.eps_start)
            else:
                epsilon = args.eps_end

            action = select_action(
                model=online_net,
                obs_vec=obs_vec,
                action_mask=current_mask,
                epsilon=epsilon,
                device=device,
                algo=args.algo,
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

            # learn
            if (
                global_step >= args.learning_starts
                and len(replay) >= args.batch_size
                and global_step % args.train_freq == 0
            ):
                for _ in range(args.gradient_steps):
                    batch = replay.sample(args.batch_size, device)
                    if args.algo == "dqn":
                        last_loss = dqn_train_step(
                            online_net=online_net,
                            target_net=target_net,
                            optimizer=optimizer,
                            batch=batch,
                            gamma=args.gamma,
                        )
                    else:
                        last_loss = qrdqn_train_step(
                            online_net=online_net,
                            target_net=target_net,
                            optimizer=optimizer,
                            batch=batch,
                            gamma=args.gamma,
                            num_quantiles=args.num_quantiles,
                            kappa=args.kappa,
                        )

            # target update
            if global_step % args.target_update_interval == 0:
                target_net.load_state_dict(online_net.state_dict())

            # periodic eval
            if global_step % args.eval_every_steps == 0:
                eval_stats = evaluate(
                    model=online_net,
                    adapter=adapter,
                    algo=args.algo,
                    device=device,
                    eval_episodes=args.eval_episodes,
                    eval_seed=args.eval_seed,
                )
                eval_row = {
                    "global_step": global_step,
                    "episode": episode_idx,
                    **eval_stats,
                }
                append_csv(eval_log_csv, eval_row)

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
                            "algo": args.algo,
                            "obs_dim": obs_dim,
                            "action_dim": action_dim,
                            "hidden_dims": hidden_dims,
                            "num_quantiles": args.num_quantiles,
                            "config": config,
                        },
                        best_ckpt_path,
                    )

        recent_returns.append(ep_return)

        train_row = {
            "global_step": global_step,
            "episode": episode_idx,
            "episode_len": ep_len,
            "episode_return": float(ep_return),
            "recent20_avg_return": float(np.mean(recent_returns)),
            "epsilon": float(epsilon),
            "loss": "" if last_loss is None else float(last_loss),
        }
        append_csv(train_log_csv, train_row)

        if (episode_idx + 1) % args.log_every_episodes == 0:
            print(
                f"[Train] episode={episode_idx+1} "
                f"step={global_step} "
                f"ep_return={ep_return:.4f} "
                f"recent20={np.mean(recent_returns):.4f} "
                f"epsilon={epsilon:.4f} "
                f"loss={last_loss}"
            )

        episode_idx += 1

    # final save
    last_ckpt_path = os.path.join(run_dir, "last_model.pt")
    torch.save(
        {
            "model_state_dict": online_net.state_dict(),
            "algo": args.algo,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": hidden_dims,
            "num_quantiles": args.num_quantiles,
            "config": config,
        },
        last_ckpt_path,
    )

    final_eval = evaluate(
        model=online_net,
        adapter=adapter,
        algo=args.algo,
        device=device,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
    )

    final_summary = {
        "run_name": args.run_name,
        "algo": args.algo,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "best_mean_return": best_mean_return,
        "final_mean_return": final_eval["mean_return"],
        "final_std_return": final_eval["std_return"],
        "elapsed_seconds": time.time() - start_time,
        "continuous_obs": args.continuous_obs,
        "sort_rows": args.sort_rows,
        "add_diff": args.add_diff,
        "hidden_dims": hidden_dims,
        "lr": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
    }

    with open(os.path.join(run_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()