# src/sixnimmt_env/opponent_policies.py
import json
import os
import random
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO


# ============================================================
# Small helpers
# ============================================================

def _resolve_model_path(path: str) -> str:
    """
    Accept both:
      - ".../best_model"
      - ".../best_model.zip"
    """
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    raise FileNotFoundError(f"Cannot find model checkpoint: {path}")


def _base_obs_to_numpy_dict(obs: dict) -> dict:
    """
    Keep exactly the same keys as env.py.
    Use numpy arrays so model.predict(...) is stable.
    """
    return {
        "player_hand": np.asarray(obs["player_hand"], dtype=np.int64),
        "last_value_of_rows": np.asarray(obs["last_value_of_rows"], dtype=np.int64),
        "length_of_rows": np.asarray(obs["length_of_rows"], dtype=np.int64),
        "table_bulls": np.asarray(obs["table_bulls"], dtype=np.int64),
    }


# ============================================================
# PPO-style dict observation preprocess
# Matches:
#   - none
#   - continuous
#   - addobservation
#   - sorted
#   - sorted_addobservation
# in scripts/train_maskable_ppo.py / scripts/eval_policy.py
# ============================================================

def preprocess_obs_for_maskable_ppo(obs: dict, wrapper_name: str) -> dict:
    base = _base_obs_to_numpy_dict(obs)

    if wrapper_name == "none":
        return base

    hand = np.asarray(obs["player_hand"], dtype=np.float32)
    tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32)
    lengths = np.asarray(obs["length_of_rows"], dtype=np.int64)
    bulls = np.asarray(obs["table_bulls"], dtype=np.float32)

    if wrapper_name == "continuous":
        return {
            "player_hand": hand / 104.0,
            "last_value_of_rows": tails / 104.0,
            "length_of_rows": lengths,
            "table_bulls": bulls / 30.0,
        }

    if wrapper_name == "addobservation":
        hand_feat = hand / 104.0
        tails_feat = tails / 104.0
        bulls_feat = bulls / 30.0
        hand_vs_rows = hand_feat[:, None] - tails_feat[None, :]
        return {
            "player_hand": hand_feat.astype(np.float32),
            "last_value_of_rows": tails_feat.astype(np.float32),
            "length_of_rows": lengths,
            "table_bulls": bulls_feat.astype(np.float32),
            "hand_vs_rows": hand_vs_rows.astype(np.float32),
        }

    if wrapper_name == "sorted":
        order = np.argsort(tails)
        tails = tails[order]
        lengths = lengths[order]
        bulls = bulls[order]
        return {
            "player_hand": (hand / 104.0).astype(np.float32),
            "last_value_of_rows": (tails / 104.0).astype(np.float32),
            "length_of_rows": lengths,
            "table_bulls": (bulls / 30.0).astype(np.float32),
        }

    if wrapper_name == "sorted_addobservation":
        order = np.argsort(tails)
        tails = tails[order]
        lengths = lengths[order]
        bulls = bulls[order]

        hand_feat = hand / 104.0
        tails_feat = tails / 104.0
        bulls_feat = bulls / 30.0
        hand_vs_rows = hand_feat[:, None] - tails_feat[None, :]

        return {
            "player_hand": hand_feat.astype(np.float32),
            "last_value_of_rows": tails_feat.astype(np.float32),
            "length_of_rows": lengths,
            "table_bulls": bulls_feat.astype(np.float32),
            "hand_vs_rows": hand_vs_rows.astype(np.float32),
        }

    raise ValueError(f"Unknown wrapper_name={wrapper_name}")


# ============================================================
# Heuristic+PPO flat observation preprocess
# Matches FlatFeatureObsWrapper in train_heuristic_rl.py / eval_heuristic_rl.py
# ============================================================

def preprocess_obs_for_heuristic_ppo(
    obs: dict,
    continuous_obs: bool = True,
    sort_rows: bool = False,
    add_diff: bool = False,
) -> np.ndarray:
    hand = np.asarray(obs["player_hand"], dtype=np.float32)         # (10,)
    tails = np.asarray(obs["last_value_of_rows"], dtype=np.float32) # (4,)
    lengths = np.asarray(obs["length_of_rows"], dtype=np.float32)   # (4,)
    bulls = np.asarray(obs["table_bulls"], dtype=np.float32)        # (4,)

    if sort_rows:
        order = np.argsort(tails)
        tails = tails[order]
        lengths = lengths[order]
        bulls = bulls[order]

    valid_hand = hand > 0

    if continuous_obs:
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

    if add_diff:
        diff = np.zeros((10, 4), dtype=np.float32)
        for i in range(10):
            if valid_hand[i]:
                diff[i] = hand[i] - tails
        if continuous_obs:
            diff = diff / 104.0
        feats.append(diff.reshape(-1))

    out = np.concatenate(feats).astype(np.float32)
    return out


# ============================================================
# Base interface
# ============================================================

class BaseOpponentPolicy:
    def reset(self, seed: Optional[int] = None):
        pass

    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        raise NotImplementedError


# ============================================================
# Simple opponents
# ============================================================

class RandomOpponentPolicy(BaseOpponentPolicy):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        valid = [i for i, ok in enumerate(action_mask) if ok]
        if not valid:
            raise ValueError("No valid action for RandomOpponentPolicy.")
        return self.rng.choice(valid)


class LowestOpponentPolicy(BaseOpponentPolicy):
    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        for i, ok in enumerate(action_mask):
            if ok:
                return i
        raise ValueError("No valid action for LowestOpponentPolicy.")


class HighestOpponentPolicy(BaseOpponentPolicy):
    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        for i in range(len(action_mask) - 1, -1, -1):
            if action_mask[i]:
                return i
        raise ValueError("No valid action for HighestOpponentPolicy.")


# ============================================================
# Frozen PPO opponent (standard PPO line)
# ============================================================

class FrozenMaskablePPOPolicy(BaseOpponentPolicy):
    """
    For weights from:
      scripts/train_maskable_ppo.py
      scripts/eval_policy.py
    """

    def __init__(
        self,
        model_path: str,
        wrapper_name: str = "none",
        device: str = "auto",
        deterministic: bool = True,
    ):
        self.model_path = _resolve_model_path(model_path)
        self.wrapper_name = wrapper_name
        self.device = device
        self.deterministic = deterministic
        self.model = MaskablePPO.load(self.model_path, device=device)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str,
        checkpoint: str = "best_model",
        device: str = "auto",
        deterministic: bool = True,
    ):
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        wrapper_name = config["wrapper"]
        model_path = os.path.join(run_dir, checkpoint)
        return cls(
            model_path=model_path,
            wrapper_name=wrapper_name,
            device=device,
            deterministic=deterministic,
        )

    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        proc_obs = preprocess_obs_for_maskable_ppo(obs, self.wrapper_name)
        action_mask = np.asarray(action_mask, dtype=bool)

        action, _ = self.model.predict(
            proc_obs,
            deterministic=self.deterministic,
            action_masks=action_mask,
        )
        return int(action)


# ============================================================
# Frozen heuristic+PPO opponent
# ============================================================

class FrozenHeuristicPPOPolicy(BaseOpponentPolicy):
    """
    For weights from:
      scripts/train_heuristic_rl.py
      scripts/eval_heuristic_rl.py
    """

    def __init__(
        self,
        model_path: str,
        continuous_obs: bool = True,
        sort_rows: bool = False,
        add_diff: bool = False,
        device: str = "auto",
        deterministic: bool = True,
    ):
        self.model_path = _resolve_model_path(model_path)
        self.continuous_obs = continuous_obs
        self.sort_rows = sort_rows
        self.add_diff = add_diff
        self.device = device
        self.deterministic = deterministic
        self.model = MaskablePPO.load(self.model_path, device=device)

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str,
        checkpoint: str = "best_finetuned_model",
        device: str = "auto",
        deterministic: bool = True,
    ):
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_path = os.path.join(run_dir, checkpoint)
        return cls(
            model_path=model_path,
            continuous_obs=bool(config["continuous_obs"]),
            sort_rows=bool(config["sort_rows"]),
            add_diff=bool(config["add_diff"]),
            device=device,
            deterministic=deterministic,
        )

    def act(self, obs: dict, action_mask, env=None, player_id: Optional[int] = None) -> int:
        proc_obs = preprocess_obs_for_heuristic_ppo(
            obs,
            continuous_obs=self.continuous_obs,
            sort_rows=self.sort_rows,
            add_diff=self.add_diff,
        )
        action_mask = np.asarray(action_mask, dtype=bool)

        action, _ = self.model.predict(
            proc_obs,
            deterministic=self.deterministic,
            action_masks=action_mask,
        )
        return int(action)