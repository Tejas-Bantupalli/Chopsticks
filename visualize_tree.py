import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from models import Player, Move, State, Game, StandardNode, WinNode, LoopNode
from game_theory import build_tree, collect_all_nodes, classify_positions_graph, pretty_status, UNKNOWN, WIN, LOSE, DRAW

def visualize_decision_tree(root_node, status_map, depth_limit=4, stop_at_win_lose=True):
    """Visualize the decision tree with limited depth to keep it readable.

    Args:
        root_node: The root node of the tree
        status_map: Map of nodes to their WIN/LOSE/DRAW status
        depth_limit: Maximum depth to explore
        stop_at_win_lose: If True, don't explore children of WIN/LOSE nodes
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []
    node_shapes = {}

    # Track node IDs
    node_id_map = {}
    next_id = [0]

    def get_node_id(node):
        # Resolve LoopNodes to their actual target
        actual_node = node.points_to if isinstance(node, LoopNode) else node
        if actual_node not in node_id_map:
            node_id_map[actual_node] = next_id[0]
            next_id[0] += 1
        return node_id_map[actual_node]

    # BFS to traverse tree with depth tracking
    queue = deque([(root_node, 0, 0, 0)])  # (node, depth, x_pos, parent_id)
    visited = set()
    x_positions = {0: [0]}  # Track x positions at each depth

    while queue:
        node, depth, x_hint, parent_id = queue.popleft()

        if depth > depth_limit:
            continue

        # Resolve LoopNodes to their actual target
        actual_node = node.points_to if isinstance(node, LoopNode) else node
        node_id = get_node_id(actual_node)

        # Determine x position
        if depth not in x_positions:
            x_positions[depth] = []

        if node_id not in pos:
            # Find next available x position at this depth
            x_pos = len(x_positions[depth])
            x_positions[depth].append(x_pos)
            pos[node_id] = (x_pos, -depth)

        # Add to graph
        if node_id not in visited:
            visited.add(node_id)

            # Create label (use actual_node)
            if isinstance(actual_node, StandardNode):
                state = actual_node.state
                label = f"{node_id}\n{state.curr_player.hands}\nvs\n{state.next_player.hands}"
                node_shapes[node_id] = 'o'  # circle
            elif isinstance(actual_node, WinNode):
                state = actual_node.state
                winner = actual_node.winner
                label = f"{node_id}\nWIN\n{winner}"
                node_shapes[node_id] = 's'  # square
            else:
                label = f"{node_id}\n?"
                node_shapes[node_id] = 'o'

            labels[node_id] = label
            G.add_node(node_id)

            # Add edge from parent
            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            # Process children - only if not a terminal node or winning/losing position
            # Don't explore beyond WinNodes or StandardNodes classified as WIN/LOSE (if enabled)
            should_explore = False
            if isinstance(actual_node, StandardNode) and depth < depth_limit:
                if stop_at_win_lose:
                    node_status = status_map.get(actual_node, UNKNOWN)
                    # Only explore if the node is DRAW or UNKNOWN, not WIN or LOSE
                    if node_status not in [WIN, LOSE]:
                        should_explore = True
                else:
                    # Explore all StandardNodes regardless of status
                    should_explore = True

            if should_explore:
                for i, (move, next_node) in enumerate(actual_node.transitions):
                    queue.append((next_node, depth + 1, x_hint + i, node_id))
        else:
            # Node already visited, but still add edge from parent
            if parent_id is not None:
                G.add_edge(parent_id, node_id)

    # Draw the graph
    plt.figure(figsize=(20, 12))

    # Draw nodes with different colors based on type
    for node_id in G.nodes():
        if node_id in node_id_map.values():
            # Reverse lookup to get actual node
            actual_node = next(k for k, v in node_id_map.items() if v == node_id)
            if isinstance(actual_node, StandardNode):
                node_colors.append('lightblue')
            elif isinstance(actual_node, WinNode):
                node_colors.append('lightgreen')
            else:
                node_colors.append('gray')

    nx.draw(G, pos, labels=labels, with_labels=True,
            node_color=node_colors, node_size=2000,
            font_size=8, font_weight='bold',
            arrows=True, arrowsize=20, arrowstyle='->')

    plt.title(f"Chopsticks Decision Tree (depth limit: {depth_limit})\nBlue=Standard, Green=Win (loops point to actual nodes)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to decision_tree_visualization.png")
    plt.close()


def visualize_with_status(root_node, status_map, depth_limit=3, stop_at_win_lose=True):
    """Visualize tree with game-theoretic status (WIN/LOSE/DRAW).

    Args:
        root_node: The root node of the tree
        status_map: Map of nodes to their WIN/LOSE/DRAW status
        depth_limit: Maximum depth to explore
        stop_at_win_lose: If True, don't explore children of WIN/LOSE nodes
    """
    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []
    edge_labels = {}

    node_id_map = {}
    next_id = [0]

    def get_node_id(node):
        # Resolve LoopNodes to their actual target
        actual_node = node.points_to if isinstance(node, LoopNode) else node
        if actual_node not in node_id_map:
            node_id_map[actual_node] = next_id[0]
            next_id[0] += 1
        return node_id_map[actual_node]

    queue = deque([(root_node, 0, None, None)])
    visited = set()
    level_counts = {}

    while queue:
        node, depth, parent_id, move = queue.popleft()

        if depth > depth_limit:
            continue

        # Resolve LoopNodes to their actual target
        actual_node = node.points_to if isinstance(node, LoopNode) else node
        node_id = get_node_id(actual_node)

        if node_id in visited:
            # Still add edge if from different parent
            if parent_id is not None:
                edge = (parent_id, node_id)
                if move:
                    edge_labels[edge] = str(move)
                G.add_edge(parent_id, node_id)
            continue

        visited.add(node_id)

        # Position nodes
        if depth not in level_counts:
            level_counts[depth] = 0
        x_pos = level_counts[depth]
        level_counts[depth] += 1
        pos[node_id] = (x_pos * 3, -depth * 2)

        # Create label with status (use actual_node)
        status = status_map.get(actual_node, "?")
        if isinstance(actual_node, StandardNode):
            state = actual_node.state
            status_str = pretty_status(status) if isinstance(status, int) else str(status)
            label = f"{state.curr_player.hands} vs {state.next_player.hands}\n[{status_str}]"
        elif isinstance(actual_node, WinNode):
            label = f"WIN: {actual_node.winner}"
        else:
            label = "?"

        labels[node_id] = label
        G.add_node(node_id)

        # Color based on status
        if isinstance(status, int):
            if status == 1:  # WIN
                node_colors.append('green')
            elif status == 2:  # LOSE
                node_colors.append('red')
            elif status == 3:  # DRAW
                node_colors.append('yellow')
            else:
                node_colors.append('lightgray')
        else:
            if isinstance(actual_node, WinNode):
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')

        # Add edge from parent
        if parent_id is not None:
            edge = (parent_id, node_id)
            if move:
                edge_labels[edge] = str(move)
            G.add_edge(parent_id, node_id)

        # Process children - only if not a terminal node or winning/losing position
        # Don't explore beyond WinNodes or StandardNodes classified as WIN/LOSE (if enabled)
        should_explore = False
        if isinstance(actual_node, StandardNode) and depth < depth_limit:
            if stop_at_win_lose:
                node_status = status_map.get(actual_node, UNKNOWN)
                # Only explore if the node is DRAW or UNKNOWN, not WIN or LOSE
                if node_status not in [WIN, LOSE]:
                    should_explore = True
            else:
                # Explore all StandardNodes regardless of status
                should_explore = True

        if should_explore:
            for move, next_node in actual_node.transitions:
                queue.append((next_node, depth + 1, node_id, move))

    # Draw
    plt.figure(figsize=(24, 14))
    nx.draw(G, pos, labels=labels, with_labels=True,
            node_color=node_colors, node_size=3000,
            font_size=7, font_weight='bold',
            arrows=True, arrowsize=15, arrowstyle='->')

    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title("Chopsticks Decision Tree with Game Status\nGreen=WIN, Red=LOSE, Yellow=DRAW, Orange=LOOP",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('decision_tree_with_status.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to decision_tree_with_status.png")
    plt.close()


if __name__ == "__main__":
    # Initialize game
    rules = Game(threshold=5, modular=False, split_rule='restrictive')
    root_state = State(Player((1, 1)), Player((1, 1)))

    print("Building game tree...")
    root_node = build_tree(root_state, rules)

    print("Collecting nodes...")
    all_nodes = collect_all_nodes(root_node)
    print(f"Total nodes: {len(all_nodes)}")

    print("Classifying positions...")
    status_map = classify_positions_graph(all_nodes)

    print("\nGenerating visualizations...")
    depth = 13
    # Generate with stop_at_win_lose=True (default - cleaner view)
    visualize_decision_tree(root_node, status_map, depth_limit=depth, stop_at_win_lose=True)
    visualize_with_status(root_node, status_map, depth_limit=depth, stop_at_win_lose=True)

    print("\nDone! Check the PNG files.")
    print("To see full tree including children of WIN/LOSE nodes, set stop_at_win_lose=False")
