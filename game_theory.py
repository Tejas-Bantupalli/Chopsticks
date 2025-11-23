from collections import defaultdict, deque
from collections import Counter
children = defaultdict(list)
visited = set()

def normalize(state):
    p0, p1, turn = state
    
    p0 = ((p0[0] % 5), (p0[1] % 5))
    p1 = ((p1[0] % 5), (p1[1] % 5))
    
    p0 = tuple(sorted(p0))
    p1 = tuple(sorted(p1))
    
    return (p0, p1, turn)


def generate_moves(state):
    p0, p1, turn = state

    me = p0 if turn else p1   #
    opp = p1 if turn else p0

    next_states = []

    for i in range(2):      
        for j in range(2):  
            if me[i] != 0 and opp[j] != 0:
                nm = list(me)
                no = list(opp)
                no[j] = (no[j] + me[i]) % 5

                if turn:
                    next_p0, next_p1 = nm, no
                else:
                    next_p0, next_p1 = no, nm

                next_states.append(
                    normalize((next_p0, next_p1, not turn))
                )

    return next_states

def build_tree(state):
    state = normalize(state)
    if state in visited:
        return

    visited.add(state)

    succ = generate_moves(state)
    children[state] = succ

    for s in succ:
        build_tree(s)

UNKNOWN, WIN, LOSE, DRAW = 0, 1, 2, 3

def classify_positions():

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
    return {WIN: "WIN", LOSE: "LOSE", DRAW: "DRAW", UNKNOWN: "UNKNOWN"}[code]

def best_moves_from(state, status):
    succs = children[state]

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
    root = ((1, 1), (1, 1), True)  

    build_tree(root)
    print("Reachable states:", len(visited))

    status = classify_positions()


    counts = Counter(status.values())
    print("Counts:", {pretty_status(k): v for k, v in counts.items()})

    print("\nStart state:", root, "=>", pretty_status(status[root]))

    print("\nBest moves from the start:")
    for s in best_moves_from(root, status):
        print("  ->", s, "which is", pretty_status(status[s]))

if __name__ == "__main__":
    main()

