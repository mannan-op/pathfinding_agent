import heapq

# Grid Size
ROWS = 25
COLS = 35

# Heuristic (Manhattan Distance)
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get valid neighbors (4-directional)
def get_neighbors(pos, walls):
    r, c = pos
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    neighbors = []

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in walls:
            neighbors.append(((nr, nc), 1))  # cost = 1

    return neighbors

# Reconstruct path
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    return path[::-1]

# A* Algorithm
def astar(start, goal, walls):
    open_heap = []
    heapq.heappush(open_heap, (manhattan(start, goal), start))

    came_from = {start: None}
    g_cost = {start: 0}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            return path, len(closed)

        for neighbor, cost in get_neighbors(current, walls):
            if neighbor in closed:
                continue

            tentative_g = g_cost[current] + cost

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_heap, (f_cost, neighbor))
                came_from[neighbor] = current

    return None, len(closed)

# Example Usage
if __name__ == "__main__":
    start = (2, 2)
    goal = (20, 30)

    # Example walls
    walls = {
        (3,2),(4,2),(5,2),(6,2),
        (6,3),(6,4),(6,5),
        (10,15),(11,15),(12,15),(13,15)
    }

    path, nodes_expanded = astar(start, goal, walls)

    if path:
        print("Path found!")
        print("Path:", path)
        print("Path cost:", len(path) - 1)
        print("Nodes expanded:", nodes_expanded)
    else:
        print("No path found.")
        print("Nodes expanded:", nodes_expanded)