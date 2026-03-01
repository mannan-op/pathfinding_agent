import heapq
import math

# Grid Size
ROWS = 25
COLS = 35

# ─── Heuristics ──────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# ─── Get Neighbors ───────────────────────────────────────
def get_neighbors(pos, walls):
    r, c = pos
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    neighbors = []

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in walls:
            neighbors.append(((nr, nc), 1))  # cost = 1

    return neighbors

# ─── Reconstruct Path ────────────────────────────────────
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]

# ─── Greedy Best-First Search ────────────────────────────
def gbfs(start, goal, walls, heuristic):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    visited = {start}
    came_from = {start: None}
    nodes_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded

        for neighbor, _ in get_neighbors(current, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(open_heap, (heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    return None, nodes_expanded

# ─── A* Search ───────────────────────────────────────────
def astar(start, goal, walls, heuristic):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    came_from = {start: None}
    g_cost = {start: 0}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded

        for neighbor, cost in get_neighbors(current, walls):
            if neighbor in closed:
                continue

            tentative_g = g_cost[current] + cost

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_cost, neighbor))
                came_from[neighbor] = current

    return None, nodes_expanded

# ─── Example Usage ───────────────────────────────────────
if __name__ == "__main__":
    start = (2, 2)
    goal = (20, 30)

    walls = {
        (3,2),(4,2),(5,2),(6,2),
        (6,3),(6,4),(6,5),
        (10,15),(11,15),(12,15),(13,15)
    }

    heuristics = {
        "Manhattan": manhattan,
        "Euclidean": euclidean
    }

    for h_name, h_func in heuristics.items():
        print(f"\n=== Using {h_name} Heuristic ===")

        print("Running GBFS...")
        path_gbfs, nodes_gbfs = gbfs(start, goal, walls, h_func)
        if path_gbfs:
            print("GBFS Path Cost:", len(path_gbfs) - 1)
        else:
            print("GBFS No Path Found")
        print("GBFS Nodes Expanded:", nodes_gbfs)

        print("\nRunning A*...")
        path_astar, nodes_astar = astar(start, goal, walls, h_func)
        if path_astar:
            print("A* Path Cost:", len(path_astar) - 1)
        else:
            print("A* No Path Found")
        print("A* Nodes Expanded:", nodes_astar)