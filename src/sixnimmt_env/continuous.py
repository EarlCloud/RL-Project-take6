import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "player_hand": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            "last_value_of_rows": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            "length_of_rows": spaces.MultiDiscrete([6] * 4),
            "table_bulls": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        })

    def observation(self, obs):

        return {
            "player_hand": np.array(obs["player_hand"], dtype=np.float32) / 104.0,
            "last_value_of_rows": np.array(obs["last_value_of_rows"], dtype=np.float32) / 104.0,
            "length_of_rows": obs["length_of_rows"],
            "table_bulls": np.array(obs["table_bulls"], dtype=np.float32) / 30.0,
        }
        

class ContinuousObservationWrapper_sorted(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "player_hand": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            "last_value_of_rows": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            "length_of_rows": spaces.MultiDiscrete([6] * 4),
            "table_bulls": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        })

    def observation(self, obs):

        last_vals = np.array(obs["last_value_of_rows"], dtype=np.float32)
        row_lengths = np.array(obs["length_of_rows"])
        bulls = np.array(obs["table_bulls"], dtype=np.float32)

        order = np.argsort(last_vals)

        last_vals = last_vals[order]
        row_lengths = row_lengths[order]
        bulls = bulls[order]

        return {
            "player_hand": np.array(obs["player_hand"], dtype=np.float32) / 104.0,
            "last_value_of_rows": last_vals / 104.0,
            "length_of_rows": row_lengths,
            "table_bulls": bulls / 30.0,
        }
        
class ContinuousObservationWrapper_addobservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "player_hand": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            "last_value_of_rows": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            "length_of_rows": spaces.MultiDiscrete([6] * 4),
            "table_bulls": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),

            # new: 10×4 difference matrix
            "hand_vs_rows": spaces.Box(-1.0, 1.0, shape=(10,4), dtype=np.float32),
        })

    def observation(self, obs):
        # normalized continuous features
        player_hand = np.array(obs["player_hand"], dtype=np.float32) / 104.0
        last_value_of_rows = np.array(obs["last_value_of_rows"], dtype=np.float32) / 104.0
        table_bulls = np.array(obs["table_bulls"], dtype=np.float32) / 30.0
        length_of_rows = obs["length_of_rows"]  # keep discrete

        # --- compute difference matrix ---
        # shape (10,4)
        hand_vs_rows = player_hand[:, None] - last_value_of_rows[None, :]  # broadcasting

        return {
            "player_hand": player_hand,
            "last_value_of_rows": last_value_of_rows,
            "length_of_rows": length_of_rows,
            "table_bulls": table_bulls,
            "hand_vs_rows": hand_vs_rows.astype(np.float32),
        }
        
class ContinuousObservationWrapper_sorted_addobservation(gym.ObservationWrapper):
    """Observation wrapper that:
       1) Sorts table rows by their last card values
       2) Normalizes continuous fields
       3) Adds hand-vs-row difference matrix
       4) Keeps length_of_rows discrete
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "player_hand": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            "last_value_of_rows": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            "length_of_rows": spaces.MultiDiscrete([6] * 4),
            "table_bulls": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            "hand_vs_rows": spaces.Box(-1.0, 1.0, shape=(10, 4), dtype=np.float32),
        })

    def observation(self, obs):
        # --- normalize continuous fields ---
        player_hand = np.array(obs["player_hand"], dtype=np.float32) / 104.0
        last_vals = np.array(obs["last_value_of_rows"], dtype=np.float32)
        row_lengths = np.array(obs["length_of_rows"])
        bulls = np.array(obs["table_bulls"], dtype=np.float32)

        # --- sort table rows by last card value ---
        order = np.argsort(last_vals)
        last_vals = last_vals[order]
        row_lengths = row_lengths[order]
        bulls = bulls[order]

        last_vals /= 104.0
        bulls /= 30.0

        # --- compute hand-vs-rows difference matrix ---
        hand_vs_rows = player_hand[:, None] - last_vals[None, :]  # shape (10,4)

        return {
            "player_hand": player_hand,
            "last_value_of_rows": last_vals,
            "length_of_rows": row_lengths,
            "table_bulls": bulls,
            "hand_vs_rows": hand_vs_rows.astype(np.float32),
        }