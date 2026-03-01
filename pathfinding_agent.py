import pygame
import heapq
import math
import random
import time
import sys

# ─── Constants ────────────────────────────────────────────────────────────────
SIDEBAR_W = 280
CELL_MIN   = 12
FPS        = 60

# Palette
C_BG       = (10,  12,  20)
C_GRID     = (25,  30,  45)
C_WALL     = (40,  48,  70)
C_START    = (80, 200, 120)
C_GOAL     = (220, 80,  80)
C_FRONTIER = (220,190,  50)
C_VISITED  = (60,  90, 160)
C_PATH     = (80, 210, 210)
C_AGENT    = (255,255,255)
C_OBSTACLE = (55,  65,  90)
C_SIDEBAR  = (16,  20,  32)
C_TEXT     = (200,210,230)
C_ACCENT   = (80, 200, 200)
C_BTN      = (30,  38,  60)
C_BTN_H    = (50,  65, 100)
C_BTN_A    = (60, 150, 150)

# ─── Heuristics ───────────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ─── Search Algorithms ────────────────────────────────────────────────────────
def get_neighbors(pos, rows, cols, walls):
    r, c = pos
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    result = []
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in walls:
            result.append(((nr,nc), 1))
    return result

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path

def gbfs(start, goal, rows, cols, walls, heuristic_fn):
    """Greedy Best-First Search — yields steps for visualization"""
    open_heap = [(heuristic_fn(start, goal), start)]
    visited   = {start}
    came_from = {start: None}
    frontier_set = {start}
    expanded  = set()
    nodes_visited = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        frontier_set.discard(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            yield {'type':'done','path':path,'nodes':nodes_visited}
            return

        expanded.add(current)
        nodes_visited += 1

        for neighbor, _ in get_neighbors(current, rows, cols, walls):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_heap, (heuristic_fn(neighbor, goal), neighbor))
                frontier_set.add(neighbor)

        yield {'type':'step','expanded':set(expanded),'frontier':set(frontier_set)}

    yield {'type':'no_path'}

def astar(start, goal, rows, cols, walls, heuristic_fn):
    """A* Search — yields steps for visualization"""
    g_cost = {start: 0}
    f_cost = {start: heuristic_fn(start, goal)}
    open_heap = [(f_cost[start], start)]
    came_from = {start: None}
    closed    = set()
    frontier_set = {start}
    nodes_visited = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        frontier_set.discard(current)
        closed.add(current)
        nodes_visited += 1

        if current == goal:
            path = reconstruct_path(came_from, goal)
            yield {'type':'done','path':path,'nodes':nodes_visited}
            return

        for neighbor, cost in get_neighbors(current, rows, cols, walls):
            if neighbor in closed:
                continue
            tentative_g = g_cost[current] + cost
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f = tentative_g + heuristic_fn(neighbor, goal)
                f_cost[neighbor] = f
                came_from[neighbor] = current
                heapq.heappush(open_heap, (f, neighbor))
                frontier_set.add(neighbor)

        yield {'type':'step','expanded':set(closed),'frontier':set(frontier_set)}

    yield {'type':'no_path'}

# ─── UI Helpers ───────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.active = active
        self.hovered = False

    def draw(self, surf, font):
        col = C_BTN_A if self.active else (C_BTN_H if self.hovered else C_BTN)
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, C_ACCENT if self.active else C_GRID, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, C_TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False

# ─── Main App ─────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1100, 700), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent — AI2002")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 12)
        self.font_m = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_t = pygame.font.SysFont("Consolas", 22, bold=True)

        self.rows   = 25
        self.cols   = 35
        self.walls  = set()
        self.start  = (2, 2)
        self.goal   = (self.rows-3, self.cols-3)

        # State
        self.algo      = 'A*'          # 'GBFS' | 'A*'
        self.heuristic = 'Manhattan'   # 'Manhattan' | 'Euclidean'
        self.mode      = 'idle'        # 'idle' | 'running' | 'done' | 'dynamic'
        self.draw_mode = 'wall'        # 'wall' | 'start' | 'goal'
        self.dynamic   = False
        self.speed     = 30            # steps/sec
        self.density   = 0.25

        self.expanded  = set()
        self.frontier  = set()
        self.path      = []
        self.agent_pos = None
        self.agent_idx = 0
        self.nodes_vis = 0
        self.path_cost = 0
        self.exec_ms   = 0
        self.msg       = ""
        self.gen       = None
        self.step_acc  = 0.0

        self._build_ui()
        self._generate_maze()

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        x = 10
        bw, bh = 260, 30
        self.btns = {
            'algo_gbfs':  Button((x,10,bw,bh),  "GBFS",       self.algo=='GBFS'),
            'algo_astar': Button((x,45,bw,bh),  "A* Search",  self.algo=='A*'),
            'h_man':      Button((x,100,bw,bh), "Manhattan",  self.heuristic=='Manhattan'),
            'h_euc':      Button((x,135,bw,bh), "Euclidean",  self.heuristic=='Euclidean'),
            'draw_wall':  Button((x,190,bw,bh), "Draw Walls", self.draw_mode=='wall'),
            'draw_start': Button((x,225,bw,bh), "Move Start", self.draw_mode=='start'),
            'draw_goal':  Button((x,260,bw,bh), "Move Goal",  self.draw_mode=='goal'),
            'gen':        Button((x,310,bw,bh), "Generate Maze"),
            'clear':      Button((x,345,bw,bh), "Clear Walls"),
            'run':        Button((x,395,bw,bh), "▶  Run Search"),
            'stop':       Button((x,430,bw,bh), "■  Stop"),
            'dynamic':    Button((x,480,bw,bh), "Dynamic Mode: OFF"),
            'speed_up':   Button((x,515,90,bh), "Faster"),
            'speed_dn':   Button((x+170,515,90,bh), "Slower"),
        }

    def _hfn(self):
        return manhattan if self.heuristic == 'Manhattan' else euclidean

    def _run_search(self, start=None, goal=None, walls=None):
        s = start or self.start
        g = goal  or self.goal
        w = walls if walls is not None else self.walls
        fn = gbfs if self.algo == 'GBFS' else astar
        t0 = time.time()
        self.gen = fn(s, g, self.rows, self.cols, w, self._hfn())
        self.exec_ms = (time.time()-t0)*1000
        self.expanded.clear()
        self.frontier.clear()
        self.path = []
        self.mode = 'running'
        self.msg  = ""

    # ── Maze generation ───────────────────────────────────────────────────────
    def _generate_maze(self):
        self.walls.clear()
        total = self.rows * self.cols
        candidates = [(r,c) for r in range(self.rows) for c in range(self.cols)
                      if (r,c) != self.start and (r,c) != self.goal]
        k = int(total * self.density)
        for cell in random.sample(candidates, min(k, len(candidates))):
            self.walls.add(cell)
        self._reset_state()

    def _reset_state(self):
        self.expanded.clear(); self.frontier.clear()
        self.path = []; self.agent_pos = None; self.agent_idx = 0
        self.nodes_vis = 0; self.path_cost = 0; self.exec_ms = 0
        self.mode = 'idle'; self.gen = None; self.msg = ""

    # ── Grid drawing ──────────────────────────────────────────────────────────
    def _cell_size(self):
        w, h = self.screen.get_size()
        grid_w = w - SIDEBAR_W
        cs = min(grid_w // self.cols, h // self.rows)
        return max(cs, CELL_MIN)

    def _cell_rect(self, r, c):
        cs = self._cell_size()
        ox = SIDEBAR_W + (self.screen.get_width()-SIDEBAR_W - self.cols*cs)//2
        oy = (self.screen.get_height() - self.rows*cs)//2
        return pygame.Rect(ox + c*cs, oy + r*cs, cs, cs)

    def _pixel_to_cell(self, px, py):
        cs = self._cell_size()
        ox = SIDEBAR_W + (self.screen.get_width()-SIDEBAR_W - self.cols*cs)//2
        oy = (self.screen.get_height() - self.rows*cs)//2
        c  = (px - ox) // cs
        r  = (py - oy) // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def _draw_grid(self):
        surf = self.screen
        cs   = self._cell_size()
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self._cell_rect(r, c)
                pos  = (r, c)
                if pos in self.walls:
                    color = C_WALL
                elif pos == self.start:
                    color = C_START
                elif pos == self.goal:
                    color = C_GOAL
                elif pos in self.path:
                    color = C_PATH
                elif pos in self.expanded:
                    color = C_VISITED
                elif pos in self.frontier:
                    color = C_FRONTIER
                else:
                    color = C_BG
                pygame.draw.rect(surf, color, rect)
                if cs > 6:
                    pygame.draw.rect(surf, C_GRID, rect, 1)

        # Agent
        if self.agent_pos:
            rect = self._cell_rect(*self.agent_pos)
            cx, cy = rect.centerx, rect.centery
            r2 = max(cs//3, 4)
            pygame.draw.circle(surf, C_AGENT, (cx, cy), r2)
            pygame.draw.circle(surf, C_ACCENT, (cx, cy), r2, 2)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _draw_sidebar(self):
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, C_SIDEBAR, (0, 0, SIDEBAR_W, h))
        pygame.draw.line(self.screen, C_GRID, (SIDEBAR_W, 0), (SIDEBAR_W, h), 2)

        # Title
        t = self.font_t.render("PathAgent", True, C_ACCENT)
        self.screen.blit(t, (10, 570+10))
        sub = self.font_s.render("AI2002 — Dynamic Search", True, C_TEXT)
        self.screen.blit(sub, (10, 570+36))

        # Section labels
        def label(txt, y):
            s = self.font_s.render(txt, True, (120,130,160))
            self.screen.blit(s, (10, y))

        label("ALGORITHM", 5)
        label("HEURISTIC", 95)
        label("EDIT MODE", 185)
        label("MAP CONTROLS", 305)
        label("SEARCH CONTROLS", 390)
        label("DYNAMIC MODE", 475)

        for btn in self.btns.values():
            btn.draw(self.screen, self.font_s)

        # Speed display
        spd_txt = self.font_s.render(f"Speed: {self.speed} steps/s", True, C_TEXT)
        self.screen.blit(spd_txt, (10+90+5, 515+8))

        # Metrics
        y = 560
        metrics = [
            ("Nodes Expanded", f"{self.nodes_vis}"),
            ("Path Cost",      f"{self.path_cost}"),
            ("Exec Time",      f"{self.exec_ms:.1f} ms"),
            ("Grid",           f"{self.rows}×{self.cols}"),
            ("Algorithm",      self.algo),
            ("Heuristic",      self.heuristic),
        ]
        # draw metrics box
        box_h = len(metrics)*18 + 10
        # draw label at y-18
        pygame.draw.rect(self.screen, C_BTN, (5, y-20, SIDEBAR_W-10, box_h+20), border_radius=6)
        label("METRICS", y-18)
        for i,(k,v) in enumerate(metrics):
            ks = self.font_s.render(k+":", True, (130,140,170))
            vs = self.font_s.render(v, True, C_ACCENT)
            self.screen.blit(ks, (10, y + i*18))
            self.screen.blit(vs, (SIDEBAR_W - vs.get_width() - 10, y + i*18))

        # Message
        if self.msg:
            ms = self.font_s.render(self.msg, True, (220,120,80))
            self.screen.blit(ms, (10, h-20))

        # Legend
        legend = [
            (C_START,    "Start"),
            (C_GOAL,     "Goal"),
            (C_WALL,     "Wall"),
            (C_FRONTIER, "Frontier"),
            (C_VISITED,  "Visited"),
            (C_PATH,     "Path"),
            (C_AGENT,    "Agent"),
        ]
        lx, ly = 10, h-160
        label("LEGEND", ly-14)
        for i, (col, name) in enumerate(legend):
            pygame.draw.rect(self.screen, col, (lx, ly+i*18+2, 12, 12), border_radius=3)
            s = self.font_s.render(name, True, C_TEXT)
            self.screen.blit(s, (lx+18, ly+i*18))

    # ── Step the search generator ─────────────────────────────────────────────
    def _advance_search(self, dt):
        if self.mode != 'running' or self.gen is None:
            return
        self.step_acc += dt * self.speed
        steps = int(self.step_acc)
        self.step_acc -= steps
        for _ in range(steps):
            try:
                result = next(self.gen)
                if result['type'] == 'step':
                    self.expanded = result['expanded']
                    self.frontier = result['frontier']
                elif result['type'] == 'done':
                    self.expanded = result.get('expanded', self.expanded)
                    self.path     = result['path']
                    self.nodes_vis = result['nodes']
                    self.path_cost = len(self.path)-1
                    self.exec_ms   = (time.time() - self._t0)*1000
                    self.mode      = 'done'
                    self.agent_pos = self.start
                    self.agent_idx = 0
                    break
                elif result['type'] == 'no_path':
                    self.msg  = "No path found!"
                    self.mode = 'idle'
                    break
            except StopIteration:
                self.mode = 'idle'
                break

    # ── Move agent along path ─────────────────────────────────────────────────
    def _move_agent(self, dt):
        if self.mode != 'done' or not self.path:
            return
        self.step_acc += dt * (self.speed/2)
        steps = int(self.step_acc)
        self.step_acc -= steps
        for _ in range(steps):
            if self.agent_idx < len(self.path):
                self.agent_pos = self.path[self.agent_idx]
                self.agent_idx += 1

                # Dynamic obstacle spawning
                if self.dynamic and random.random() < 0.05:
                    # Spawn obstacle not on critical cells
                    r = random.randint(0, self.rows-1)
                    c = random.randint(0, self.cols-1)
                    pos = (r, c)
                    if pos not in (self.start, self.goal, self.agent_pos) and pos not in self.walls:
                        self.walls.add(pos)
                        # Check if obstacle is on remaining path
                        remaining = self.path[self.agent_idx:]
                        if pos in remaining:
                            # Re-plan from agent's current position
                            self._replan()
                            return

    def _replan(self):
        self.expanded.clear(); self.frontier.clear()
        self._t0 = time.time()
        fn = gbfs if self.algo == 'GBFS' else astar
        self.gen = fn(self.agent_pos, self.goal, self.rows, self.cols, self.walls, self._hfn())
        self.mode = 'running'
        self.step_acc = 0.0

    # ── Event handling ────────────────────────────────────────────────────────
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._run_now()
                if event.key == pygame.K_c:
                    self._reset_state(); self.walls.clear()
                if event.key == pygame.K_ESCAPE:
                    self._reset_state()

            # Button handling
            for key, btn in self.btns.items():
                if btn.handle(event):
                    self._btn_action(key)

            # Grid editing
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                if pygame.mouse.get_pressed()[0]:
                    cell = self._pixel_to_cell(*pygame.mouse.get_pos())
                    if cell:
                        if self.draw_mode == 'wall':
                            if cell != self.start and cell != self.goal:
                                self.walls.add(cell)
                                self._reset_state()
                        elif self.draw_mode == 'start':
                            if cell not in self.walls and cell != self.goal:
                                self.start = cell
                                self._reset_state()
                        elif self.draw_mode == 'goal':
                            if cell not in self.walls and cell != self.start:
                                self.goal = cell
                                self._reset_state()
                if pygame.mouse.get_pressed()[2]:
                    cell = self._pixel_to_cell(*pygame.mouse.get_pos())
                    if cell and cell in self.walls:
                        self.walls.discard(cell)
                        self._reset_state()

    def _btn_action(self, key):
        if key == 'algo_gbfs':
            self.algo = 'GBFS'
            self.btns['algo_gbfs'].active = True
            self.btns['algo_astar'].active = False
        elif key == 'algo_astar':
            self.algo = 'A*'
            self.btns['algo_astar'].active = True
            self.btns['algo_gbfs'].active = False
        elif key == 'h_man':
            self.heuristic = 'Manhattan'
            self.btns['h_man'].active = True
            self.btns['h_euc'].active = False
        elif key == 'h_euc':
            self.heuristic = 'Euclidean'
            self.btns['h_euc'].active = True
            self.btns['h_man'].active = False
        elif key == 'draw_wall':
            self.draw_mode = 'wall'
            for k in ('draw_wall','draw_start','draw_goal'):
                self.btns[k].active = (k == 'draw_wall')
        elif key == 'draw_start':
            self.draw_mode = 'start'
            for k in ('draw_wall','draw_start','draw_goal'):
                self.btns[k].active = (k == 'draw_start')
        elif key == 'draw_goal':
            self.draw_mode = 'goal'
            for k in ('draw_wall','draw_start','draw_goal'):
                self.btns[k].active = (k == 'draw_goal')
        elif key == 'gen':
            self._generate_maze()
        elif key == 'clear':
            self.walls.clear(); self._reset_state()
        elif key == 'run':
            self._run_now()
        elif key == 'stop':
            self._reset_state()
        elif key == 'dynamic':
            self.dynamic = not self.dynamic
            self.btns['dynamic'].label = f"Dynamic Mode: {'ON' if self.dynamic else 'OFF'}"
            self.btns['dynamic'].active = self.dynamic
        elif key == 'speed_up':
            self.speed = min(self.speed + 5, 200)
        elif key == 'speed_dn':
            self.speed = max(self.speed - 5, 1)

    def _run_now(self):
        self._reset_state()
        self._t0 = time.time()
        self._run_search()

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self._handle_events()

            if self.mode == 'running':
                self._advance_search(dt)
            elif self.mode == 'done':
                self._move_agent(dt)

            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_sidebar()
            pygame.display.flip()


if __name__ == '__main__':
    App().run()
