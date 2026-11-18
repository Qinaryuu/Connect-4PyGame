import math
import random
import time
from collections import deque
from board import (
    ROWS,
    COLS,
    EMPTY,
    PLAYER_PIECE,
    AI_PIECE,
    copy_board,
    get_valid_locations,
    get_next_open_row,
    drop_piece,
    winning_move,
    is_terminal_node,
)


def distance_from_center(col):
    return abs((COLS // 2) - col)


def evaluate_board_static(board, piece):
    """Fungsi heuristik statis (tanpa adaptasi Bayesian) untuk agen pembanding."""
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    score = 0

    center_array = [board[r][COLS // 2] for r in range(ROWS)]
    score += center_array.count(piece) * 4

    def eval_win(window):
        s = 0
        if window.count(piece) == 4:
            s += 10000
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            s += 10
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            s += 3
        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            s -= 12
        return s

    for r in range(ROWS):
        for c in range(COLS - 3):
            score += eval_win([board[r][c + i] for i in range(4)])
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += eval_win([board[r + i][c] for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += eval_win([board[r + i][c + i] for i in range(4)])
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            score += eval_win([board[r - i][c + i] for i in range(4)])

    return score


class BaseAgent:
    def __init__(self, name, piece=PLAYER_PIECE):
        self.name = name
        self.piece = piece
        self.opp_piece = AI_PIECE if piece == PLAYER_PIECE else PLAYER_PIECE
        self.last_metrics = {
            "decision_source": name,
            "nodes_evaluated": 0,
            "pruned_branches": 0,
            "pruning_ratio": 0.0,
            "execution_time_ms": 0.0,
        }

    def choose_move(self, board):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Agen 1: Memilih kolom valid secara acak."""

    def __init__(self, piece=PLAYER_PIECE):
        super().__init__("Random Agent", piece)

    def choose_move(self, board):
        start = time.perf_counter()
        valid_cols = get_valid_locations(board)
        col = random.choice(valid_cols)
        self.last_metrics["nodes_evaluated"] = 1
        self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return col


class RuleBasedAgent(BaseAgent):
    """Agen 2: Agen aturan dasar murni (Forward Chaining tanpa pencarian pohon)."""

    def __init__(self, piece=PLAYER_PIECE):
        super().__init__("Rule-Based Agent", piece)

    def choose_move(self, board):
        start = time.perf_counter()
        valid_cols = get_valid_locations(board)
        nodes = 0

        # Rule 1: Instant Win
        for col in valid_cols:
            nodes += 1
            row = get_next_open_row(board, col)
            temp = copy_board(board)
            drop_piece(temp, row, col, self.piece)
            if winning_move(temp, self.piece):
                self.last_metrics["nodes_evaluated"] = nodes
                self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
                return col

        # Rule 2: Instant Block
        for col in valid_cols:
            nodes += 1
            row = get_next_open_row(board, col)
            temp = copy_board(board)
            drop_piece(temp, row, col, self.opp_piece)
            if winning_move(temp, self.opp_piece):
                self.last_metrics["nodes_evaluated"] = nodes
                self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
                return col

        # Fallback: pilih kolom yang paling dekat ke tengah
        col = min(valid_cols, key=distance_from_center)
        self.last_metrics["nodes_evaluated"] = nodes
        self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return col


class BFSAgent(BaseAgent):
    """Agen 3: Breadth-First Search (FIFO queue, depth-limited)."""

    def __init__(self, max_depth=3, piece=PLAYER_PIECE):
        super().__init__(f"BFS Agent (d={max_depth})", piece)
        self.max_depth = max_depth

    def choose_move(self, board):
        start = time.perf_counter()
        valid_cols = get_valid_locations(board)
        queue = deque()
        nodes = 0

        for col in valid_cols:
            row = get_next_open_row(board, col)
            b_copy = copy_board(board)
            drop_piece(b_copy, row, col, self.piece)
            queue.append((b_copy, col, 1, False))

        best_col = valid_cols[0]
        best_score = -math.inf

        while queue:
            curr_board, first_col, depth, is_my_turn = queue.popleft()
            nodes += 1

            if winning_move(curr_board, self.piece):
                best_col = first_col
                break

            if depth >= self.max_depth or is_terminal_node(curr_board):
                sc = evaluate_board_static(curr_board, self.piece)
                if sc > best_score:
                    best_score = sc
                    best_col = first_col
                continue

            next_piece = self.piece if is_my_turn else self.opp_piece
            for c in get_valid_locations(curr_board):
                r = get_next_open_row(curr_board, c)
                nxt = copy_board(curr_board)
                drop_piece(nxt, r, c, next_piece)
                queue.append((nxt, first_col, depth + 1, not is_my_turn))

        self.last_metrics["nodes_evaluated"] = nodes
        self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return best_col


class StandardMinimaxAgent(BaseAgent):
    """Agen 4: Minimax murni TANPA Alpha-Beta Pruning."""

    def __init__(self, depth=4, piece=PLAYER_PIECE):
        super().__init__(f"Minimax Murni (d={depth})", piece)
        self.depth = depth

    def _minimax_no_pruning(self, board, depth, maximizing):
        self.last_metrics["nodes_evaluated"] += 1
        valid_cols = get_valid_locations(board)
        terminal = is_terminal_node(board)

        if depth == 0 or terminal:
            if terminal:
                if winning_move(board, self.piece):
                    return (None, 1000000 + depth)
                elif winning_move(board, self.opp_piece):
                    return (None, -1000000 - depth)
                return (None, 0)
            return (None, evaluate_board_static(board, self.piece))

        if maximizing:
            value = -math.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                row = get_next_open_row(board, col)
                b_copy = copy_board(board)
                drop_piece(b_copy, row, col, self.piece)
                _, sc = self._minimax_no_pruning(b_copy, depth - 1, False)
                if sc > value:
                    value = sc
                    best_col = col
            return best_col, value
        else:
            value = math.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                row = get_next_open_row(board, col)
                b_copy = copy_board(board)
                drop_piece(b_copy, row, col, self.opp_piece)
                _, sc = self._minimax_no_pruning(b_copy, depth - 1, True)
                if sc < value:
                    value = sc
                    best_col = col
            return best_col, value

    def choose_move(self, board):
        start = time.perf_counter()
        self.last_metrics["nodes_evaluated"] = 0
        best_col, _ = self._minimax_no_pruning(board, self.depth, True)
        self.last_metrics["execution_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return best_col