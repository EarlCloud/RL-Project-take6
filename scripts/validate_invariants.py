import itertools


def _is_strictly_increasing(values):
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def validate_invariants(env):
    """
    Structural + rule invariants check for V0.

    Raises AssertionError with helpful messages when something is wrong.
    """
    table = env.table
    deck = env.deck
    players = env.players

    # -------- 1) Table rows must be non-empty --------
    assert len(table.rows) == 4, f"Expected 4 rows, got {len(table.rows)}"
    for i, row in enumerate(table.rows):
        assert len(row) >= 1, f"Row {i} is empty (should never happen)."

    # -------- 2) Row lengths must match & be within [1,5] --------
    for i, row in enumerate(table.rows):
        assert table.row_lengths[i] == len(row), (
            f"Row length mismatch at row {i}: "
            f"row_lengths={table.row_lengths[i]} vs len(row)={len(row)}"
        )
        assert 1 <= table.row_lengths[i] <= 5, (
            f"Row {i} has invalid length {table.row_lengths[i]} (expected 1..5)."
        )

    # -------- 3) Row bulls must match sum of bulls in that row --------
    for i, row in enumerate(table.rows):
        bulls_sum = sum(c.bulls for c in row)
        assert table.row_bulls[i] == bulls_sum, (
            f"Row bulls mismatch at row {i}: "
            f"row_bulls={table.row_bulls[i]} vs sum(bulls)={bulls_sum}"
        )

    # -------- 4) Each row should be strictly increasing in card values --------
    # This is a key rule property of 6 nimmt rows.
    for i, row in enumerate(table.rows):
        vals = [c.value for c in row]
        assert _is_strictly_increasing(vals), (
            f"Row {i} is not strictly increasing: {vals}. "
            f"This indicates a rule/placement bug."
        )

    # -------- 5) Global card conservation + uniqueness --------
    deck_vals = [c.value for c in deck.cards]
    hand_vals = list(itertools.chain.from_iterable([c.value for c in p.hand] for p in players))
    table_vals = list(itertools.chain.from_iterable([c.value for c in row] for row in table.rows))

    taken_vals = list(itertools.chain.from_iterable([c.value for c in p.taken] for p in players))

    all_vals = deck_vals + hand_vals + table_vals + taken_vals

    assert len(all_vals) == 104, (
        f"Card conservation broken: total={len(all_vals)} != 104. "
        f"deck={len(deck_vals)}, hands={len(hand_vals)}, table={len(table_vals)}, taken={len(taken_vals)}"
    )

    assert len(set(all_vals)) == 104, (
        "Duplicate card detected across deck/hand/table/taken (should never happen)."
    )
    
    assert all(1 <= v <= 104 for v in all_vals), (
        "Found invalid card value outside 1..104 in deck/hand/table."
    )
    # -------- 6) Action mask sanity (optional but useful) --------
    mask = env.action_masks()
    assert len(mask) == 10, f"action_masks length should be 10, got {len(mask)}"
    assert sum(mask) == len(players[0].hand), (
        f"Mask count mismatch: sum(mask)={sum(mask)} vs hand_size={len(players[0].hand)}"
    )