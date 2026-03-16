import random


# ============================================================
# Card + Deck
# ============================================================

class Card:
    def __init__(self, value: int):
        self.value = value

        # Bulls count
        if value == 55:
            self.bulls = 7
        elif value % 10 == 0:
            self.bulls = 3
        elif value % 5 == 0:
            self.bulls = 2
        elif value % 11 == 0:
            self.bulls = 5
        else:
            self.bulls = 1


class Deck:
    def __init__(self):
        self.cards = [Card(i) for i in range(1, 105)]

    def shuffle(self, rng=None):
        if rng is None:
            import random as _random
            _random.shuffle(self.cards)
        else:
            rng.shuffle(self.cards)

# ============================================================
# Table
# ============================================================

class Table:
    def __init__(self):
        self.rows = [[] for _ in range(4)]
        self.row_bulls = [0 for _ in range(4)]
        self.row_lengths = [0 for _ in range(4)]

    def init_deal(self, deck: Deck):
        for i in range(4):
            card = deck.cards.pop(0)
            self.rows[i].append(card)
            self.row_bulls[i] = card.bulls
            self.row_lengths[i] = 1

    def add_card_to_row(self, card: Card, row_index: int):
        """
        Normal placement (card value should be > row tail).
        If row already has 5 cards, player eats those 5 and row resets with [card].
        Returns: (eaten_bulls: int, eaten_cards: list[Card])
        """
        # Row full (already 5) -> eat them, reset with card
        if self.row_lengths[row_index] >= 5:
            eaten_cards = self.rows[row_index]
            eaten_bulls = self.row_bulls[row_index]

            self.rows[row_index] = [card]
            self.row_bulls[row_index] = card.bulls
            self.row_lengths[row_index] = 1

            return eaten_bulls, eaten_cards

        # Normal add
        self.rows[row_index].append(card)
        self.row_bulls[row_index] += card.bulls
        self.row_lengths[row_index] += 1

        return 0, []

    def force_take_row_and_replace(self, card: Card, row_index: int):
        """
        Forced eat rule:
        - player takes ALL cards currently in chosen row (length 1..5)
        - then the played card becomes the new first card of that row
        Returns: (eaten_bulls: int, eaten_cards: list[Card])
        """
        eaten_cards = self.rows[row_index]
        eaten_bulls = self.row_bulls[row_index]

        self.rows[row_index] = [card]
        self.row_bulls[row_index] = card.bulls
        self.row_lengths[row_index] = 1

        return eaten_bulls, eaten_cards

    def get_forced_row(self, card: Card) -> int:
        last_values = [row[-1].value for row in self.rows]
        if card.value < min(last_values):
            return -1
        valid_tails = [v for v in last_values if v < card.value]
        best_tail = max(valid_tails)
        return last_values.index(best_tail)

# ============================================================
# Player + Enemy
# ============================================================

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand = []
        self.score = 0
        self.taken = []  # NEW: cards eaten/taken by this player (for conservation/debug)

    def receive_card(self, card: Card):
        self.hand.append(card)

    def sort_hand(self):
        self.hand.sort(key=lambda c: c.value)

    def play_card(self, index: int) -> Card:
        return self.hand.pop(index)
class EnemyPlayer(Player):
    def __init__(self, player_id: int, strategy: str = "random", rng=None):
        super().__init__(player_id)
        self.strategy = strategy
        self.rng = rng  # NEW

    def choose_card(self, table_state=None) -> int:
        if self.strategy == "random":
            return self.rng.randint(0, len(self.hand) - 1) if self.rng else __import__("random").randint(0, len(self.hand) - 1)
        elif self.strategy == "lowest":
            return 0
        elif self.strategy == "highest":
            return len(self.hand) - 1
        else:
            return self.rng.randint(0, len(self.hand) - 1) if self.rng else __import__("random").randint(0, len(self.hand) - 1)