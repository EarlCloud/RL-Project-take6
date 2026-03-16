# scripts/eval_qrdqn_row_deepsets.py
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

from sixnimmt_env import SixQuiPrendEnv


# ============================================================
# Utilities
# ============================================================

def parse_hidden_dims(hidden_dims_str):
    if isinstance(hidden_dims_str, list):
        return hidden_dims_str
    return [int(x.strip()) for x in hidden_dims_str.split(",") if x.strip()]


# ============================================================
# Same adapter / model as training
# ============================================================

class RowDeepSetsObservationAdapter:
    def __init__(self):
        self.hand_dim = 10
        self.hand_valid_dim = 10
        self.row_dim = 4 * 3
        self.obs_dim = self.hand_dim + self.hand_valid_dim + self.row_dim

    def preprocess(self, obs: dict) -> np.ndarray:
        hand = np.asarray(obs["player_hand"], dtype=np.float32)
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

        self.phi_row = build_mlp(
            input_dim=3,
            hidden_dims=list(row_phi_hidden_dims),
            output_dim=row_embed_dim,
        )

        head_input_dim = 10 + 10 + row_embed_dim + row_embed_dim
        self.head = build_mlp(
            input_dim=head_input_dim,
            hidden_dims=head_hidden_dims,
            output_dim=action_dim * num_quantiles,
        )

    def forward(self, x):
        hand_vals = x[:, :10]
        hand_valid = x[:, 10:20]
        row_feats_flat = x[:, 20:32]
        row_feats = row_feats_flat.view(-1, 4, 3)

        row_emb = self.phi_row(row_feats.view(-1, 3))
        row_emb = row_emb.view(-1, 4, self.row_embed_dim)

        row_mean = row_emb.mean(dim=1)
        row_max = row_emb.max(dim=1).values

        head_input = torch.cat([hand_vals, hand_valid, row_mean, row_max], dim=1)
        out = self.head(head_input)
        return out.view(-1, self.action_dim, self.num_quantiles)


@torch.no_grad()
def select_greedy_action(model, obs_vec: np.ndarray, action_mask: np.ndarray, device: torch.device):
    obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = model(obs_t).mean(dim=-1).squeeze(0)
    q_values = q_values.clone()
    q_values[~torch.tensor(action_mask, dtype=torch.bool, device=device)] = -1e9
    return int(torch.argmax(q_values).item())


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_seed", type=int, default=30000)
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

    adapter = RowDeepSetsObservationAdapter()
    obs_dim = adapter.obs_dim
    action_dim = 10

    hidden_dims = config["hidden_dims"]
    row_phi_hidden_dims = config["row_phi_hidden_dims"]

    model = RowDeepSetsQRDQNNet(
        obs_dim=obs_dim,
        action_dim=action_dim,
        head_hidden_dims=hidden_dims,
        num_quantiles=config["num_quantiles"],
        row_embed_dim=config["row_embed_dim"],
        row_phi_hidden_dims=row_phi_hidden_dims,
    ).to(device)

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
            action = select_greedy_action(model, obs_vec, mask, device=device)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.asarray(returns, dtype=np.float32)
    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
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