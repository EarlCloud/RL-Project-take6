# scripts/eval_value_based.py
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

from sixnimmt_env import SixQuiPrendEnv


# ============================================================
# Same adapter / nets as training
# ============================================================

class ObservationAdapter:
    def __init__(self, continuous_obs=True, sort_rows=False, add_diff=False):
        self.continuous_obs = continuous_obs
        self.sort_rows = sort_rows
        self.add_diff = add_diff

        self.base_dim = 10 + 4 + 4 + 4
        self.diff_dim = 10 * 4 if add_diff else 0
        self.obs_dim = self.base_dim + self.diff_dim

    def preprocess(self, obs: dict) -> np.ndarray:
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
        assert out.shape[0] == self.obs_dim
        return out


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
        return self.net(x)


class QRDQNNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims, num_quantiles: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.net = build_mlp(obs_dim, hidden_dims, action_dim * num_quantiles)

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.action_dim, self.num_quantiles)


@torch.no_grad()
def select_greedy_action(model, obs_vec: np.ndarray, action_mask: np.ndarray, algo: str, device: torch.device):
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    if algo == "dqn":
        q_values = model(obs_t).squeeze(0)
    elif algo == "qrdqn":
        q_values = model(obs_t).mean(dim=-1).squeeze(0)
    else:
        raise ValueError(algo)

    q_values = q_values.clone()
    q_values[~torch.tensor(action_mask, dtype=torch.bool, device=device)] = -1e9
    return int(torch.argmax(q_values).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_seed", type=int, default=20000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config_path = os.path.join(args.run_dir, "config.json")
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    adapter = ObservationAdapter(
        continuous_obs=config["continuous_obs"],
        sort_rows=config["sort_rows"],
        add_diff=config["add_diff"],
    )

    algo = config["algo"]
    hidden_dims = config["hidden_dims"]
    action_dim = 10
    obs_dim = adapter.obs_dim

    if algo == "dqn":
        model = DQNNet(obs_dim, action_dim, hidden_dims).to(device)
    elif algo == "qrdqn":
        model = QRDQNNet(obs_dim, action_dim, hidden_dims, config["num_quantiles"]).to(device)
    else:
        raise ValueError(algo)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = SixQuiPrendEnv()
    returns = []

    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.eval_seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_vec = adapter.preprocess(obs)
            mask = np.array(env.action_masks(), dtype=bool)
            action = select_greedy_action(model, obs_vec, mask, algo=algo, device=device)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.array(returns, dtype=np.float32)
    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "algo": algo,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
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