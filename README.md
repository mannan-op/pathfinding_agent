# Dynamic Pathfinding Agent — AI2002 Assignment 2

A real-time pathfinding visualizer implementing **Greedy Best-First Search (GBFS)** and **A\* Search** with dynamic obstacle support on a grid-based environment.

---

## Features

- **Algorithms:** Greedy Best-First Search, A* Search
- **Heuristics:** Manhattan Distance, Euclidean Distance
- **Interactive Map Editor:** Draw/erase walls by clicking
- **Random Maze Generator:** User-defined obstacle density
- **Dynamic Mode:** Obstacles spawn mid-navigation; agent re-plans in real-time
- **Live Metrics:** Nodes expanded, path cost, execution time
- **Visualization:** Frontier (yellow), Visited (blue), Path (cyan), Agent (white)

---

## Requirements

- Python 3.8+
- Pygame

## Installation

```bash
# Clone the repository
git clone https://github.com/F223618/AI2002-Assignment2.git
cd AI2002-Assignment2

# Install dependencies
pip install pygame
```

## Running the Application

```bash
python pathfinding_agent.py
```

---

## Controls

| Action | Control |
|--------|---------|
| Run Search | `R` key or **▶ Run Search** button |
| Stop / Reset | `Esc` key or **■ Stop** button |
| Generate Maze | `G` key or **Generate Maze** button |
| Clear Walls | `C` key or **Clear Walls** button |
| Draw Wall | Left-click on grid (Draw Walls mode) |
| Erase Wall | Right-click on grid |
| Move Start/Goal | Select mode, then left-click on grid |

---

## Project Structure

```
AI2002-Assignment2/
│
├── pathfinding_agent.py      # Main application (Pygame GUI)
├── README.md                 # This file
└── Report_AI2002_A2.pdf      # Comprehensive project report
```

---

## Algorithm Overview

### Greedy Best-First Search (GBFS)
- Uses only the heuristic: `f(n) = h(n)`
- Fast but **not guaranteed to find the optimal path**
- Uses a Strict Visited List to prevent cycles

### A* Search
- Uses combined cost: `f(n) = g(n) + h(n)`
- **Guaranteed optimal** with an admissible, consistent heuristic
- Uses an Expanded List (allows node re-opening for optimality)

### Heuristics
- **Manhattan Distance:** `|x1-x2| + |y1-y2|` — admissible for 4-directional grid
- **Euclidean Distance:** `sqrt((x1-x2)^2 + (y1-y2)^2)` — admissible, tighter estimate

---

## Dynamic Re-planning

When **Dynamic Mode** is enabled:
1. Obstacles spawn randomly each step with a small probability
2. If an obstacle appears on the agent's current planned path, the agent immediately re-runs the selected search algorithm from its current position
3. If the obstacle is not on the current path, no re-planning occurs (efficient)

---

## Dependencies

```
pygame>=2.0.0
```

Install with:
```bash
pip install pygame
```

---

## Author

**[Your Name]**  
Roll No: 22F-3618 
AI2002 — Artificial Intelligence (Spring 2026)  
National University of Computer & Emerging Sciences, Chiniot-Faisalabad Campus

