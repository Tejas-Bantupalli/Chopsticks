from collections import defaultdict, deque, Counter
from models import Player, Move, Game

children = defaultdict(list)
visited = set()


def generate_possible_moves(game):
    """Generate all possible Move objects from the current game state."""
    moves = []
    me = game.curr_player

    # Generate all possible attack moves
    for i in range(2):
        for j in range(2):
            moves.append(Move.attack(i, j))

    # Generate split moves based on split rule
    total_fingers = sum(me.hands)

    if game.split_rule == 'restrictive':
        # Only split from 4-0 or 2-0
        if me.hands in [(4, 0), (2, 0)]:
            k = me.hands[0] // 2
            moves.append(Move.split((k, k)))
    else:
        # Generate all possible splits that preserve total fingers
        for i in range(total_fingers + 1):
            j = total_fingers - i
            if i <= j:  # Avoid duplicates since Player normalizes
                moves.append(Move.split((j, i)))

    return moves


def generate_moves(game):
    """Generate all legal next game states from the current game."""
    possible_moves = generate_possible_moves(game)
    next_games = []

    for move in possible_moves:
        try:
            next_game = move.apply(game)
            next_games.append(next_game)
        except ValueError:
            # Move is illegal, skip it
            pass

    # Deduplicate
    seen = set()
    unique_games = []
    for g in next_games:
        if g not in seen:
            seen.add(g)
            unique_games.append(g)

    return unique_games


def build_tree(game):
    """Build the game tree from a starting game state."""
    if game in visited:
        return

    visited.add(game)
    succ = generate_moves(game)
    children[game] = succ

    for s in succ:
        build_tree(s)


UNKNOWN, WIN, LOSE, DRAW = 0, 1, 2, 3


def classify_positions():
    """Classify each game position as WIN, LOSE, or DRAW."""
    parents = defaultdict(set)
    for s, succs in children.items():
        for t in succs:
            parents[t].add(s)

    status = {s: UNKNOWN for s in visited}
    remaining_children = {s: len(children[s]) for s in visited}

    q = deque()
    for s, succs in children.items():
        if not succs:
            status[s] = LOSE
            q.append(s)

    while q:
        s = q.popleft()
        st = status[s]

        for p in parents.get(s, []):
            if status[p] != UNKNOWN:
                continue

            if st == LOSE:
                status[p] = WIN
                q.append(p)

            elif st == WIN:
                remaining_children[p] -= 1
                if remaining_children[p] == 0:
                    status[p] = LOSE
                    q.append(p)

    for s in status:
        if status[s] == UNKNOWN:
            status[s] = DRAW

    return status


def pretty_status(code):
    """Convert status code to string."""
    return {WIN: "WIN", LOSE: "LOSE", DRAW: "DRAW", UNKNOWN: "UNKNOWN"}[code]


def best_moves_from(game, status):
    """Find the best moves from a given game state."""
    succs = children[game]

    winning_moves = []
    drawing_moves = []

    for s in succs:
        if status[s] == LOSE:
            winning_moves.append(s)
        elif status[s] == DRAW:
            drawing_moves.append(s)

    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    if winning_moves:
        return dedup(winning_moves)
    if drawing_moves:
        return dedup(drawing_moves)
    return dedup(succs)


def main():
    root = Game(Player((1, 1)), Player((1, 1)))

    build_tree(root)
    print("Reachable states:", len(visited))

    status = classify_positions()

    counts = Counter(status.values())
    print("Counts:", {pretty_status(k): v for k, v in counts.items()})

    print(f"\nStart state: {root} => {pretty_status(status[root])}")

    print("\nBest moves from the start:")
    for s in best_moves_from(root, status):
        print(f"  -> {s} which is {pretty_status(status[s])}")


if __name__ == "__main__":
    main()
