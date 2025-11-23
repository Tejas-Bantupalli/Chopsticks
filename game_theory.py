from collections import defaultdict, deque, Counter
from models import Player, Move, State, Game, StandardNode, WinNode, LoopNode

# Map from State to its StandardNode (for detecting loops)
state_to_node = {}


def generate_possible_moves(state, rules):
    """Generate all possible Move objects from the current state."""
    moves = []
    me = state.curr_player

    # Generate all possible attack moves
    for i in range(2):
        for j in range(2):
            moves.append(Move.attack(i, j))

    # Generate split moves based on split rule
    total_fingers = sum(me.hands)

    if rules.split_rule == 'restrictive':
        # Only split from 4-0 or 2-0
        if me.hands in [(4, 0), (2, 0)]:
            k = me.hands[0] // 2
            moves.append(Move.split((k, k)))
    else:
        # Generate all possible splits that preserve total fingers
        for i in range(total_fingers + 1):
            j = total_fingers - i
            if i <= j and i != me.hands[0]:  # Avoid duplicates or no-ops
                moves.append(Move.split((j, i)))

    return moves


def generate_moves(state, rules):
    """Generate all legal moves and their resulting states."""
    possible_moves = generate_possible_moves(state, rules)
    move_state_pairs = []

    for move in possible_moves:
        try:
            next_state = move.apply(state, rules)
            move_state_pairs.append((move, next_state))
        except ValueError:
            # Move is illegal, skip it
            pass

    # Deduplicate states (but keep the moves)
    seen_states = {}
    unique_pairs = []
    for move, next_state in move_state_pairs:
        if next_state not in seen_states:
            seen_states[next_state] = move
            unique_pairs.append((move, next_state))

    return unique_pairs


def build_tree(state, rules):
    """Build the game tree from a starting state using the new node structure."""
    # Check if this state already has a node
    if state in state_to_node:
        # Return a LoopNode pointing to the existing node
        return LoopNode(state_to_node[state])

    # Check if this is a terminal state
    if state.is_terminal():
        return WinNode(state, state.winner())

    # Create a StandardNode for this state
    node = StandardNode(state)
    state_to_node[state] = node

    # Generate all possible moves and next states
    move_state_pairs = generate_moves(state, rules)

    # Recursively build the tree for each next state
    for move, next_state in move_state_pairs:
        next_node = build_tree(next_state, rules)
        node.add_transition(move, next_node)

    return node


UNKNOWN, WIN, LOSE, DRAW = 0, 1, 2, 3


def classify_positions_graph(all_nodes):
    """Classify all nodes in the graph using iterative backward induction."""
    # Build parent relationships
    parents = defaultdict(set)
    for node in all_nodes:
        if isinstance(node, StandardNode):
            for _move, next_node in node.transitions:
                # Resolve LoopNodes to their actual targets
                target = next_node.points_to if isinstance(next_node, LoopNode) else next_node
                parents[target].add(node)

    # Initialize status map
    status = {node: UNKNOWN for node in all_nodes if isinstance(node, StandardNode)}
    remaining_children = {}

    for node in all_nodes:
        if isinstance(node, StandardNode):
            # Count children, resolving LoopNodes
            child_count = 0
            for _move, next_node in node.transitions:
                target = next_node.points_to if isinstance(next_node, LoopNode) else next_node
                if isinstance(target, StandardNode):
                    child_count += 1
            remaining_children[node] = child_count

    # Start with terminal nodes
    q = deque()
    for node in all_nodes:
        if isinstance(node, WinNode):
            # Winner already won, so current player (who would move) has lost
            status[node] = LOSE
            q.append(node)
        elif isinstance(node, StandardNode) and not node.transitions:
            # No moves available is a loss
            status[node] = LOSE
            q.append(node)

    # Backward induction
    while q:
        node = q.popleft()
        st = status[node]

        for parent in parents.get(node, []):
            if status[parent] != UNKNOWN:
                continue

            if st == LOSE:
                # Parent can win by moving to this losing state
                status[parent] = WIN
                q.append(parent)
            elif st == WIN:
                # Parent's opponent wins, so one option eliminated
                remaining_children[parent] -= 1
                if remaining_children[parent] == 0:
                    # All moves lead to opponent winning
                    status[parent] = LOSE
                    q.append(parent)

    # Mark remaining UNKNOWN nodes as DRAW
    for node in status:
        if status[node] == UNKNOWN:
            status[node] = DRAW

    # Also map LoopNodes to their target's status
    for node in all_nodes:
        if isinstance(node, LoopNode):
            status[node] = status[node.points_to]

    return status


def pretty_status(code):
    """Convert status code to string."""
    return {WIN: "WIN", LOSE: "LOSE", DRAW: "DRAW", UNKNOWN: "UNKNOWN"}[code]


def collect_all_nodes(root_node, visited=None):
    """Collect all unique nodes in the tree."""
    if visited is None:
        visited = set()

    if root_node in visited:
        return visited

    visited.add(root_node)

    if isinstance(root_node, StandardNode):
        for move, next_node in root_node.transitions:
            collect_all_nodes(next_node, visited)

    return visited


def best_moves_from(node, status_map):
    """Find the best moves from a given node."""
    if not isinstance(node, StandardNode):
        return []

    winning_moves = []
    drawing_moves = []
    other_moves = []

    for move, next_node in node.transitions:
        next_status = status_map.get(next_node, UNKNOWN)

        if next_status == LOSE:
            winning_moves.append((move, next_node))
        elif next_status == DRAW:
            drawing_moves.append((move, next_node))
        else:
            other_moves.append((move, next_node))

    if winning_moves:
        return winning_moves
    if drawing_moves:
        return drawing_moves
    return other_moves


def main():
    # Initialize game rules
    rules = Game(threshold=5, modular=False, split_rule='restrictive')

    # Initialize starting state
    root_state = State(Player((1, 1)), Player((1, 1)))

    # Build the game tree
    root_node = build_tree(root_state, rules)

    # Collect all nodes
    all_nodes = collect_all_nodes(root_node)
    print(f"Total nodes in graph: {len(all_nodes)}")

    # Count node types
    standard_count = sum(1 for n in all_nodes if isinstance(n, StandardNode))
    win_count = sum(1 for n in all_nodes if isinstance(n, WinNode))
    loop_count = sum(1 for n in all_nodes if isinstance(n, LoopNode))
    print(f"StandardNode: {standard_count}, WinNode: {win_count}, LoopNode: {loop_count}")

    # Classify positions
    status_map = classify_positions_graph(all_nodes)

    # Count statuses for StandardNodes
    status_counts = Counter()
    for node in all_nodes:
        if isinstance(node, StandardNode):
            status_counts[status_map.get(node, UNKNOWN)] += 1

    print("Counts:", {pretty_status(k): v for k, v in status_counts.items()})

    print(f"\nStart state: {root_state} => {pretty_status(status_map[root_node])}")

    print("\nBest moves from the start:")
    for move, next_node in best_moves_from(root_node, status_map):
        next_state = next_node.points_to.state if isinstance(next_node, LoopNode) else next_node.state
        print(f"  -> {move} to {next_state} which is {pretty_status(status_map.get(next_node, UNKNOWN))}")


if __name__ == "__main__":
    main()
