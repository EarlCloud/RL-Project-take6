# scripts/eval_old_model_on_fixedopp_env.py
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

from sixnimmt_env import SixQuiPrendEnvFixedOpponents
from sixnimmt_env.opponent_policies import (
    FrozenHeuristicPPOPolicy,
    FrozenMaskablePPOPolicy,
)


def parse_hidden_dims(x):
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        return [int(v.strip()) for v in x.split(",") if v.strip()]
    raise ValueError(f"Unsupported hidden_dims format: {type(x)}")


def get_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class RowDeepSetsObservationAdapter:
    def __init__(self):
        self.obs_dim = 32

    def preprocess(self, obs: dict) -> np.ndarray:
        hand = np.asarray(obs["player_hand"], dtype=np.float32)
        hand_valid = (hand > 0).astype(np.float32)

        tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32) / 104.0
        lengths = np.asarray(obs["length_of_rows"], dtype=np.float32) / 5.0
        bulls = np.asarray(obs["table_bulls"], dtype=np.float32) / 30.0

        hand_vals = hand / 104.0
        row_feats = np.stack([tails, lengths, bulls], axis=1).reshape(-1).astype(np.float32)

        x = np.concatenate([hand_vals, hand_valid, row_feats]).astype(np.float32)
        assert x.shape[0] == 32
        return x


def build_mlp(input_dim: int, hidden_dims, output_dim: int):
    layers = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, output_dim))
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
    q = model(obs_t).mean(dim=-1).squeeze(0)
    q = q.clone()
    q[~torch.tensor(action_mask, dtype=torch.bool, device=device)] = -1e9
    return int(torch.argmax(q).item())


def build_fixed_opponents(args):
    ppo1 = FrozenMaskablePPOPolicy.from_run_dir(
        run_dir=args.ppo1_run_dir,
        checkpoint=args.ppo_checkpoint,
        device=args.opponent_device,
        deterministic=True,
    )
    ppo2 = FrozenMaskablePPOPolicy.from_run_dir(
        run_dir=args.ppo2_run_dir,
        checkpoint=args.ppo_checkpoint,
        device=args.opponent_device,
        deterministic=True,
    )
    heur = FrozenHeuristicPPOPolicy.from_run_dir(
        run_dir=args.heuristic_run_dir,
        checkpoint=args.heuristic_checkpoint,
        device=args.opponent_device,
        deterministic=True,
    )
    return [ppo1, ppo2, heur]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default="checkpoints/outputs/qrdqn_rowdeepsets_seed4",
    )
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument("--eval_seed", type=int, default=30000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--opponent_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--ppo1_run_dir", type=str, default="checkpoints/frozen_opponents/ppo_seed1")
    parser.add_argument("--ppo2_run_dir", type=str, default="checkpoints/frozen_opponents/ppo_seed2")
    parser.add_argument("--heuristic_run_dir", type=str, default="checkpoints/frozen_opponents/heuristic_seed1")
    parser.add_argument("--ppo_checkpoint", type=str, default="best_model")
    parser.add_argument("--heuristic_checkpoint", type=str, default="best_finetuned_model")

    args = parser.parse_args()
    device = get_device(args.device)

    config_path = os.path.join(args.run_dir, "config.json")
    ckpt_path = os.path.join(args.run_dir, args.checkpoint)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    hidden_dims = parse_hidden_dims(config["hidden_dims"])
    row_phi_hidden_dims = parse_hidden_dims(config.get("row_phi_hidden_dims", [64, 64]))
    row_embed_dim = int(config.get("row_embed_dim", 64))
    num_quantiles = int(config["num_quantiles"])

    adapter = RowDeepSetsObservationAdapter()

    model = RowDeepSetsQRDQNNet(
        obs_dim=adapter.obs_dim,
        action_dim=10,
        head_hidden_dims=hidden_dims,
        num_quantiles=num_quantiles,
        row_embed_dim=row_embed_dim,
        row_phi_hidden_dims=row_phi_hidden_dims,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    opponent_policies = build_fixed_opponents(args)
    env = SixQuiPrendEnvFixedOpponents(
        opponent_policies=opponent_policies,
        opponent_name="ppo1_ppo2_heur1",
    )

    returns = []

    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.eval_seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_vec = adapter.preprocess(obs)
            mask = np.array(env.action_masks(), dtype=bool)
            action = select_greedy_action(model, obs_vec, mask, device)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    returns = np.asarray(returns, dtype=np.float32)

    summary = {
        "run_dir": args.run_dir,
        "checkpoint": args.checkpoint,
        "test_env": "fixed_opponents_env",
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_penalty": float(-returns.mean()),
        "ppo1_run_dir": args.ppo1_run_dir,
        "ppo2_run_dir": args.ppo2_run_dir,
        "heuristic_run_dir": args.heuristic_run_dir,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    json_out = os.path.join(args.run_dir, "cross_eval_old_model_on_fixedopp_env.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_out = os.path.join(args.run_dir, "cross_eval_old_model_on_fixedopp_env_episode_returns.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return"])
        writer.writeheader()
        for i, r in enumerate(returns):
            writer.writerow({"episode": i, "return": float(r)})


if __name__ == "__main__":
    main()