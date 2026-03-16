import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .core import Deck, Table, Player, EnemyPlayer


class SixQuiPrendEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.rng = None
        # --- Game objects ---
        self.deck = None
        self.table = None
        self.players = None

        # Action space: agent chooses only which card to play (index in sorted hand)
        self.action_space = spaces.Discrete(10)

        # Observation space
        self.observation_space = spaces.Dict({
            "player_hand": spaces.MultiDiscrete([105] * 10),
            "last_value_of_rows": spaces.MultiDiscrete([105] * 4),
            "length_of_rows": spaces.MultiDiscrete([6] * 4),
            "table_bulls": spaces.MultiDiscrete([30] * 4),
        })

    # ========================================================
    # Reset
    # ========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # NEW: create env-local RNG
        self.rng = random.Random(seed)

        # New deck + shuffle with rng
        self.deck = Deck()
        self.deck.shuffle(rng=self.rng)

        # New table
        self.table = Table()

        # Players (agent = player 0)
        self.players = [
            Player(0),
            EnemyPlayer(1, rng=self.rng),
            EnemyPlayer(2, rng=self.rng),
            EnemyPlayer(3, rng=self.rng),
        ]

        # Deal 10 cards each
        for _ in range(10):
            for p in self.players:
                p.receive_card(self.deck.cards.pop(0))

        for p in self.players:
            p.sort_hand()

        self.table.init_deal(self.deck)

        return self._get_observation(), {} 

    # ========================================================
    # Observation helper
    # ========================================================
    def _get_observation(self):
        # Agent hand padded with 0
        hand_vals = [c.value for c in self.players[0].hand]
        hand_vals += [0] * (10 - len(hand_vals))

        return {
            "player_hand": hand_vals,
            "last_value_of_rows": [row[-1].value for row in self.table.rows],
            "length_of_rows": self.table.row_lengths,
            "table_bulls": self.table.row_bulls,
        }

    # ========================================================
    # Greedy forced row eating (V0: eat-min)
    # ========================================================
    def choose_best_row_to_eat(self):
        min_bulls = min(self.table.row_bulls)
        candidates = [i for i, b in enumerate(self.table.row_bulls) if b == min_bulls]
        return self.rng.choice(candidates)
    # ========================================================
    # Action mask (V0: provide mask, sampling later can use it)
    # ========================================================
    def action_masks(self):
        hand_size = len(self.players[0].hand)
        return [i < hand_size for i in range(10)]

    # ========================================================
    # Step
    # ========================================================
    def step(self, action):
        agent = self.players[0]
        hand_size = len(agent.hand)

        if isinstance(action, np.integer):
            action = int(action)

        assert isinstance(action, int), f"Action must be int, got {type(action)}"
        assert 0 <= action < hand_size, (
            f"Invalid action={action}. hand_size={hand_size}. "
            f"Valid action indices are [0..{hand_size-1}]. "
            f"action_mask={self.action_masks()}"
        ) 
        agent_penalty = 0

        # --- 1) Everyone plays a card ---
        played_cards = []

        agent_card = agent.play_card(action)
        played_cards.append((agent, agent_card))

        for enemy in self.players[1:]:
            idx = enemy.choose_card()
            enemy_card = enemy.play_card(idx)
            played_cards.append((enemy, enemy_card))

        # --- 2) Sort by card value ---
        played_cards.sort(key=lambda x: x[1].value)

        # --- 3) Resolve placements with correct forced-eat handling ---
        for pl, card in played_cards:
            forced_row = self.table.get_forced_row(card)

            if forced_row != -1:
                row_to_use = forced_row
                eaten_bulls, eaten_cards = self.table.add_card_to_row(card, row_to_use)
            else:
                # V0: forced eat-min row, then replace row with this card
                row_to_use = self.choose_best_row_to_eat()
                eaten_bulls, eaten_cards = self.table.force_take_row_and_replace(card, row_to_use)

            # Update score and taken pile (for conservation/debug)
            pl.score += eaten_bulls
            pl.taken.extend(eaten_cards)

            if pl.player_id == 0:
                agent_penalty += eaten_bulls

        obs = self._get_observation()
        reward = -agent_penalty
        terminated = len(agent.hand) == 0
        truncated = False

        return obs, reward, terminated, truncated, {} 

    # ========================================================
    # Render
    # ========================================================
    def render(self):
        print("\n--- TABLE ---")
        for i, row in enumerate(self.table.rows):
            vals = [c.value for c in row]
            print(f"Row {i}: {vals} | Bulls={self.table.row_bulls[i]}")

        print("\n--- AGENT HAND ---")
        print([c.value for c in self.players[0].hand])

        print("\n--- SCORES ---")
        for p in self.players:
            print(f"Player {p.player_id}: {p.score}")