# src/sixnimmt_env/env_fixed_opponents.py
import random

import numpy as np

from .core import Deck, Table, Player
from .env import SixQuiPrendEnv
from .opponent_policies import RandomOpponentPolicy


class SixQuiPrendEnvFixedOpponents(SixQuiPrendEnv):
    """
    New environment:
    - player0 is the trainable agent
    - player1~3 are frozen policies
    - reward stays exactly the same as old env: reward = -agent_penalty

    Important:
    All 4 players decide actions from the SAME round-start snapshot,
    then cards are revealed together and resolved in ascending card value.
    """

    def __init__(self, opponent_policies=None, opponent_name="fixed_opponents"):
        super().__init__()

        if opponent_policies is None:
            opponent_policies = [
                RandomOpponentPolicy(),
                RandomOpponentPolicy(),
                RandomOpponentPolicy(),
            ]

        if len(opponent_policies) != 3:
            raise ValueError(
                f"opponent_policies must have length 3, got {len(opponent_policies)}"
            )

        self.opponent_policies = opponent_policies
        self.opponent_name = opponent_name

    # ========================================================
    # Reset
    # ========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # env-local RNG, same style as old env
        self.rng = random.Random(seed)

        # deck + table
        self.deck = Deck()
        self.deck.shuffle(rng=self.rng)

        self.table = Table()

        # all 4 are plain Player objects
        self.players = [
            Player(0),
            Player(1),
            Player(2),
            Player(3),
        ]

        # deal 10 cards each
        for _ in range(10):
            for p in self.players:
                p.receive_card(self.deck.cards.pop(0))

        for p in self.players:
            p.sort_hand()

        self.table.init_deal(self.deck)

        # reset opponent policies (for random seed or future internal state)
        for i, policy in enumerate(self.opponent_policies):
            if hasattr(policy, "reset"):
                policy.reset(None if seed is None else seed + 1000 + i)

        return self._get_observation(), {}

    # ========================================================
    # Observation helpers
    # ========================================================
    def _get_observation_for_player(self, player_id: int):
        hand_vals = [c.value for c in self.players[player_id].hand]
        hand_vals += [0] * (10 - len(hand_vals))

        return {
            "player_hand": hand_vals,
            "last_value_of_rows": [row[-1].value for row in self.table.rows],
            "length_of_rows": list(self.table.row_lengths),
            "table_bulls": list(self.table.row_bulls),
        }

    def _get_observation(self):
        return self._get_observation_for_player(0)

    # ========================================================
    # Action masks
    # ========================================================
    def action_masks_for_player(self, player_id: int):
        hand_size = len(self.players[player_id].hand)
        return [i < hand_size for i in range(10)]

    def action_masks(self):
        return self.action_masks_for_player(0)

    # ========================================================
    # Internal validation
    # ========================================================
    def _validate_action(self, action, player_id: int):
        hand_size = len(self.players[player_id].hand)

        if isinstance(action, np.integer):
            action = int(action)

        assert isinstance(action, int), (
            f"Action for player {player_id} must be int, got {type(action)}"
        )
        assert 0 <= action < hand_size, (
            f"Invalid action={action} for player={player_id}. "
            f"hand_size={hand_size}. "
            f"Valid action indices are [0..{hand_size - 1}]. "
            f"action_mask={self.action_masks_for_player(player_id)}"
        )
        return action

    # ========================================================
    # Step
    # ========================================================
    def step(self, action):
        # 0) validate agent action first
        action = self._validate_action(action, player_id=0)
        agent = self.players[0]
        agent_penalty = 0

        # 1) snapshot all players' observations and masks
        #    IMPORTANT: all players decide from the same round-start state
        snapshot_obs = [self._get_observation_for_player(pid) for pid in range(4)]
        snapshot_masks = [self.action_masks_for_player(pid) for pid in range(4)]

        # 2) get actions for all players
        chosen_actions = [None] * 4
        chosen_actions[0] = action

        for pid in (1, 2, 3):
            policy = self.opponent_policies[pid - 1]
            opp_action = policy.act(
                obs=snapshot_obs[pid],
                action_mask=snapshot_masks[pid],
                env=self,
                player_id=pid,
            )
            opp_action = self._validate_action(opp_action, player_id=pid)
            chosen_actions[pid] = opp_action

        # 3) everyone now actually plays a card
        played_cards = []
        for pid in range(4):
            pl = self.players[pid]
            card = pl.play_card(chosen_actions[pid])
            played_cards.append((pl, card))

        # 4) sort by card value
        played_cards.sort(key=lambda x: x[1].value)

        # 5) resolve placements exactly like old env
        for pl, card in played_cards:
            forced_row = self.table.get_forced_row(card)

            if forced_row != -1:
                row_to_use = forced_row
                eaten_bulls, eaten_cards = self.table.add_card_to_row(card, row_to_use)
            else:
                # same V0 simplification: forced eat-min row, then replace
                row_to_use = self.choose_best_row_to_eat()
                eaten_bulls, eaten_cards = self.table.force_take_row_and_replace(card, row_to_use)

            pl.score += eaten_bulls
            pl.taken.extend(eaten_cards)

            if pl.player_id == 0:
                agent_penalty += eaten_bulls

        obs = self._get_observation()
        reward = -agent_penalty
        terminated = len(agent.hand) == 0
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info