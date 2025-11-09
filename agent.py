import os
import uuid
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

# ============================================================================
# HEURISTIC AGENT MVP CONFIGURATION
# ============================================================================
TICK_BUDGET_MS = 15
SEED = 1337
H, W = 18, 20

DIRS = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}
ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]  # tie-break priority (CW)

def torus(x, y):
    """Normalize position to torus coordinates."""
    return (x % W, y % H)

def next_pos(head, dir_name, steps=1):
    """Calculate next position after moving in direction for given steps."""
    dx, dy = DIRS[dir_name]
    x, y = head
    for _ in range(steps):
        x, y = torus(x + dx, y + dy)
    return (x, y)

def get_blocked_cells():
    """Get all blocked cells from board grid state."""
    blocked = set()
    for y in range(H):
        for x in range(W):
            if GLOBAL_GAME.board.grid[y][x] != 0:
                blocked.add((x, y))
    return blocked

def is_free(cell, blocked, my_head=None, opp_head=None):
    """Check if a cell is free.
    
    A cell is free if:
    - It's not in the blocked set
    - OR it's one of the head positions (heads can move out of their position)
    
    Args:
        cell: Position to check
        blocked: Set of blocked cells
        my_head: Optional current head position (treated as free)
        opp_head: Optional opponent head position (treated as free)
    """
    if cell in blocked:
        # Heads are allowed to move out of their current position
        if my_head and cell == my_head:
            return True
        if opp_head and cell == opp_head:
            return True
        return False
    return True

def flood_area(start, blocked, cap=200, treat_as_free=None):
    """Bounded flood-fill to measure mobility from a starting position.
    
    Args:
        start: Starting position
        blocked: Set of blocked cells
        cap: Maximum cells to explore (early stop)
        treat_as_free: Set of positions to treat as free even if in blocked
    """
    if treat_as_free is None:
        treat_as_free = set()
    
    if start in blocked and start not in treat_as_free:
        return 0
    q = deque([start])
    seen = {start}
    count = 0
    while q and count < cap:
        x, y = q.popleft()
        count += 1
        for dx, dy in DIRS.values():
            nx, ny = torus(x + dx, y + dy)
            if (nx, ny) not in seen:
                # Cell is free if not blocked OR in treat_as_free set
                if (nx, ny) not in blocked or (nx, ny) in treat_as_free:
                    seen.add((nx, ny))
                    q.append((nx, ny))
    return count

def voronoi_margin(my_next, opp_head, blocked, cap=150):
    """Dual-BFS Voronoi to compute territory difference.
    
    Head positions are always treated as free for their respective BFS seeds,
    even if they appear in blocked set or board state.
    
    Args:
        my_next: My next position
        opp_head: Opponent's head position
        blocked: Set of blocked cells
        cap: Maximum cells to explore per player (performance optimization)
    """
    q = deque()
    seen = {}
    
    # Always add starting positions (heads are free for their owner)
    q.append((my_next, 1))  # player 1
    seen[my_next] = 1
    
    q.append((opp_head, 2))  # player 2
    seen[opp_head] = 2
    
    my_cells = 1
    opp_cells = 1
    
    # Cap total exploration to avoid timeout
    total_explored = 0
    max_total = cap * 2
    
    while q and total_explored < max_total:
        (x, y), player = q.popleft()
        total_explored += 1
        
        for dx, dy in DIRS.values():
            nx, ny = torus(x + dx, y + dy)
            if (nx, ny) not in blocked and (nx, ny) not in seen:
                seen[(nx, ny)] = player
                q.append(((nx, ny), player))
                if player == 1:
                    my_cells += 1
                else:
                    opp_cells += 1
    
    return my_cells - opp_cells

def predict_opponent_next(opp_head, opp_agent, blocked, my_head, opp_head_pos):
    """Predict opponent's next move (1-ply).
    
    Uses agent.direction from engine and simulated blocked state.
    
    Args:
        opp_head: Opponent's current head position
        opp_agent: Opponent agent object (for direction)
        blocked: Current blocked cells set
        my_head: My head position (treated as free)
        opp_head_pos: Opponent head position (treated as free)
    """
    # Get opponent's current direction from engine
    opp_dir = None
    if hasattr(opp_agent, 'direction') and opp_agent.direction:
        # Map Direction enum to string
        dir_map = {
            Direction.UP: "UP",
            Direction.DOWN: "DOWN",
            Direction.LEFT: "LEFT",
            Direction.RIGHT: "RIGHT"
        }
        opp_dir = dir_map.get(opp_agent.direction)
    
    # Try to continue in current direction if legal
    if opp_dir:
        opp_next = next_pos(opp_head, opp_dir, 1)
        if is_free(opp_next, blocked, my_head, opp_head_pos):
            return opp_next
    
    # Otherwise, pick best legal move by flood area
    best_move = None
    best_area = -1
    
    for dir_name in ORDER:
        opp_next = next_pos(opp_head, dir_name, 1)
        if is_free(opp_next, blocked, my_head, opp_head_pos):
            # Treat both heads as free for mobility calculation
            area = flood_area(opp_next, blocked, cap=200, treat_as_free={my_head, opp_head_pos})
            if area > best_area:
                best_area = area
                best_move = opp_next
    
    return best_move

def compute_neighbor_degree(pos, blocked, treat_as_free=None):
    """Count number of free neighbors around a position.
    
    Args:
        pos: Position to check
        blocked: Set of blocked cells
        treat_as_free: Set of positions to treat as free even if in blocked
    """
    if treat_as_free is None:
        treat_as_free = set()
    
    degree = 0
    for dx, dy in DIRS.values():
        nx, ny = torus(pos[0] + dx, pos[1] + dy)
        if (nx, ny) not in blocked or (nx, ny) in treat_as_free:
            degree += 1
    return degree

def evaluate_move(dir_name, my_head, my_trail_set, opp_head, opp_trail_set, 
                  tick, my_dir, boosts_remaining, opp_agent, use_boost=False):
    """Evaluate a candidate move and return its score.
    
    Simulates the move and incorporates opponent's predicted next position
    for accurate world modeling.
    """
    steps = 2 if use_boost else 1
    
    # Get blocked cells from board grid (includes all AGENT-marked cells)
    blocked = get_blocked_cells()
    
    # Check if move is legal (both steps if boost)
    legal = True
    current_pos = my_head
    
    for step in range(steps):
        next = next_pos(current_pos, dir_name, 1)
        if not is_free(next, blocked, my_head, opp_head):
            legal = False
            break
        current_pos = next
    
    if not legal:
        return -1000  # Invalid move
    
    final_pos = current_pos
    
    # Compute features with simulated world state
    # Update blocked set with our new positions
    new_blocked = blocked.copy()
    temp_pos = my_head
    for step in range(steps):
        temp_pos = next_pos(temp_pos, dir_name, 1)
        new_blocked.add(temp_pos)
    
    # Predict opponent's next move and add to blocked (1-ply simulation)
    opp_pred_next = predict_opponent_next(opp_head, opp_agent, blocked, my_head, opp_head)
    if opp_pred_next:
        new_blocked.add(opp_pred_next)
    
    # Heads should be treated as free for mobility calculations
    heads_free = {my_head, opp_head}
    
    # Mobility: flood area from our final position
    area = flood_area(final_pos, new_blocked, cap=200, treat_as_free=heads_free)
    
    # Territory: Voronoi margin (capped at 150 per player)
    vor = voronoi_margin(final_pos, opp_head, new_blocked, cap=150)
    
    # Self-trap detection (treat heads as free for degree calculation)
    my_degree = compute_neighbor_degree(final_pos, new_blocked, treat_as_free=heads_free)
    self_trapped = 1 if (my_degree <= 1 and area < 8) else 0
    
    # Opp-trap detection (use opponent's predicted position if available)
    opp_eval_pos = opp_pred_next if opp_pred_next else opp_head
    opp_degree = compute_neighbor_degree(opp_eval_pos, new_blocked, treat_as_free=heads_free)
    opp_area = flood_area(opp_eval_pos, new_blocked, cap=200, treat_as_free=heads_free)
    opp_trapped = 1 if (opp_degree <= 1 and opp_area < 8) else 0
    
    # Head-on collision risk
    head_on_pen = 1 if (opp_pred_next and final_pos == opp_pred_next) else 0
    
    # Forward preference (continue current direction)
    forward_pref = 0.2 if dir_name == my_dir else 0.0
    
    # Vor weight changes with game phase
    vor_weight = 0.5 if tick < 15 else 2.5
    
    # Compute score
    score = (100 * 1 +  # alive (already filtered)
             2.0 * area +
             vor_weight * vor +
             6.0 * opp_trapped -
             5.0 * self_trapped -
             8.0 * head_on_pen +
             forward_pref)
    
    return score

def decide_best_move(player_number):
    """Main decision logic for the heuristic agent."""
    start_time = time.perf_counter()
    
    with game_lock:
        tick = GLOBAL_GAME.turns
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        
        my_head = my_agent.trail[-1] if my_agent.trail else (0, 0)
        opp_head = opp_agent.trail[-1] if opp_agent.trail else (0, 0)
        
        my_trail_set = set(my_agent.trail)
        opp_trail_set = set(opp_agent.trail)
        
        boosts_remaining = my_agent.boosts_remaining
        
        # Get current direction from engine
        my_dir = None
        if hasattr(my_agent, 'direction') and my_agent.direction:
            # Map Direction enum to string
            dir_map = {
                Direction.UP: "UP",
                Direction.DOWN: "DOWN",
                Direction.LEFT: "LEFT",
                Direction.RIGHT: "RIGHT"
            }
            my_dir = dir_map.get(my_agent.direction)
    
    # Evaluate all candidate moves
    best_move = ORDER[0]
    best_score = -float('inf')
    best_use_boost = False
    
    for dir_name in ORDER:
        # Check time budget
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > TICK_BUDGET_MS:
            break
        
        # Evaluate 1-step move
        score_1 = evaluate_move(dir_name, my_head, my_trail_set, opp_head, 
                               opp_trail_set, tick, my_dir, boosts_remaining,
                               opp_agent, use_boost=False)
        
        if score_1 > best_score:
            best_score = score_1
            best_move = dir_name
            best_use_boost = False
        
        # Evaluate 2-step boost move if we have boosts
        if boosts_remaining > 0:
            score_2 = evaluate_move(dir_name, my_head, my_trail_set, opp_head, 
                                   opp_trail_set, tick, my_dir, boosts_remaining,
                                   opp_agent, use_boost=True)
            
            # Boost policy: use boost if it gives significant advantage
            use_boost_for_this = False
            
            # Corridor escape: check if we're in tight spot
            if score_2 > score_1 + 4:
                use_boost_for_this = True
            
            # Late game with positive voronoi
            if tick >= 120 and score_2 > score_1:
                use_boost_for_this = True
            
            # Keep at least 1 boost for endgame unless corridor escape
            if boosts_remaining == 1 and tick < 150 and not (score_2 > score_1 + 10):
                use_boost_for_this = False
            
            if use_boost_for_this and score_2 > best_score:
                best_score = score_2
                best_move = dir_name
                best_use_boost = True
    
    # Format move string
    if best_use_boost:
        return f"{best_move}:BOOST"
    else:
        return best_move


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    # Use the heuristic agent to decide the best move
    move = decide_best_move(player_number)

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
