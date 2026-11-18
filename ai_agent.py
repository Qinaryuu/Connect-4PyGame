import math
import time
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


class Connect4HybridAgent:
    def __init__(self, depth=5):
        self.depth = depth
        # Prior probabilitas gaya bermain lawan (50% Agresif, 50% Defensif)
        self.p_aggressive = 0.5
        self.w_attack = 1.0
        self.w_defense = 1.0

        # Metrik performa per langkah
        self.last_metrics = {
            "decision_source": "",
            "nodes_evaluated": 0,
            "pruned_branches": 0,
            "total_branches_considered": 0,
            "pruning_ratio": 0.0,
            "execution_time_ms": 0.0,
            "p_aggressive": 0.5,
        }

    # ---------------------------------------------------------
    # 1. MODUL PENALARAN BAYESIAN
    # ---------------------------------------------------------
    def _classify_opponent_evidence(self, prev_board, opp_col):
        """Mengamati bukti (Evidence) dari langkah terakhir lawan."""
        row = get_next_open_row(prev_board, opp_col)
        if row is None:
            return "NEUTRAL"

        # Cek apakah langkah lawan memblokir ancaman 3-sebaris AI
        test_ai = copy_board(prev_board)
        drop_piece(test_ai, row, opp_col, AI_PIECE)
        if winning_move(test_ai, AI_PIECE):
            return "DEFENSIVE_BLOCK"

        # Cek apakah lawan membangun serangan sendiri atau menguasai kolom tengah
        test_opp = copy_board(prev_board)
        drop_piece(test_opp, row, opp_col, PLAYER_PIECE)
        if winning_move(test_opp, PLAYER_PIECE) or opp_col == COLS // 2:
            return "OFFENSIVE_PLAY"

        return "NEUTRAL"

    def update_bayesian_belief(self, prev_board, opp_col):
        """Memperbarui P(Agresif | Evidence) dengan Teorema Bayes & kalibrasi bobot."""
        evidence = self._classify_opponent_evidence(prev_board, opp_col)

        # Conditional Probability Table (CPT): P(E | Gaya Lawan)
        cpt = {
            "OFFENSIVE_PLAY": (0.75, 0.25),   # P(E|Agresif), P(E|Defensif)
            "DEFENSIVE_BLOCK": (0.25, 0.80),
            "NEUTRAL": (0.50, 0.50),
        }
        p_e_given_agg, p_e_given_def = cpt[evidence]
        p_def = 1.0 - self.p_aggressive

        numerator = p_e_given_agg * self.p_aggressive
        denominator = numerator + (p_e_given_def * p_def)

        if denominator > 0:
            posterior = numerator / denominator
            # Clamp [0.1, 0.9] agar keyakinan tetap adaptif sepanjang permainan
            self.p_aggressive = max(0.1, min(0.9, posterior))

        # Kalibrasi bobot heuristik secara dinamis
        self.w_defense = round(0.8 + 0.7 * self.p_aggressive, 3)
        self.w_attack = round(1.5 - 0.7 * self.p_aggressive, 3)

    # ---------------------------------------------------------
    # 2. MODUL FORWARD CHAINING / RULE-BASED
    # ---------------------------------------------------------
    def forward_chaining(self, board):
        """Mengevaluasi aturan kritis: Rule 1 (Instant Win), Rule 2 (Instant Block)."""
        valid_cols = get_valid_locations(board)

        # Rule 1: IF ada langkah yang membuat AI menang langsung -> THEN ambil langkah tsb
        for col in valid_cols:
            row = get_next_open_row(board, col)
            temp_board = copy_board(board)
            drop_piece(temp_board, row, col, AI_PIECE)
            if winning_move(temp_board, AI_PIECE):
                return col, "FORWARD_CHAINING (INSTANT_WIN)"

        # Rule 2: IF lawan bisa menang di giliran berikutnya -> THEN blokir kolom tsb
        for col in valid_cols:
            row = get_next_open_row(board, col)
            temp_board = copy_board(board)
            drop_piece(temp_board, row, col, PLAYER_PIECE)
            if winning_move(temp_board, PLAYER_PIECE):
                return col, "FORWARD_CHAINING (INSTANT_BLOCK)"

        return None, None

    # ---------------------------------------------------------
    # 3. FUNGSI EVALUASI HEURISTIK ADAPTIF
    # ---------------------------------------------------------
    def _evaluate_window(self, window):
        score = 0
        ai_count = window.count(AI_PIECE)
        opp_count = window.count(PLAYER_PIECE)
        empty_count = window.count(EMPTY)

        # Skor ofensif AI (dikalikan bobot adaptif w_attack)
        if ai_count == 4:
            score += 10000 * self.w_attack
        elif ai_count == 3 and empty_count == 1:
            score += 10 * self.w_attack
        elif ai_count == 2 and empty_count == 2:
            score += 3 * self.w_attack

        # Penalti ancaman lawan (dikalikan bobot adaptif w_defense)
        if opp_count == 3 and empty_count == 1:
            score -= 12 * self.w_defense
        elif opp_count == 2 and empty_count == 2:
            score -= 3 * self.w_defense

        return score

    def score_position(self, board):
        score = 0

        # Prioritas kontrol kolom tengah
        center_col = COLS // 2
        center_array = [board[r][center_col] for r in range(ROWS)]
        score += center_array.count(AI_PIECE) * 4 * self.w_attack

        # Evaluasi Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = [board[r][c + i] for i in range(4)]
                score += self._evaluate_window(window)

        # Evaluasi Vertikal
        for c in range(COLS):
            for r in range(ROWS - 3):
                window = [board[r + i][c] for i in range(4)]
                score += self._evaluate_window(window)

        # Evaluasi Diagonal Positif (/)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += self._evaluate_window(window)

        # Evaluasi Diagonal Negatif (\)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                window = [board[r - i][c + i] for i in range(4)]
                score += self._evaluate_window(window)

        return score

    # ---------------------------------------------------------
    # 4. MINIMAX + ALPHA-BETA PRUNING + MOVE ORDERING
    # ---------------------------------------------------------
    def _distance_from_center(self, col):
        return abs((COLS // 2) - col)

    def _order_moves(self, valid_cols):
        """Move Ordering: urutkan dari kolom tengah [3, 2, 4, 1, 5, 0, 6]."""
        return sorted(valid_cols, key=self._distance_from_center)

    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        self.last_metrics["nodes_evaluated"] += 1
        valid_cols = self._order_moves(get_valid_locations(board))
        terminal = is_terminal_node(board)

        if depth == 0 or terminal:
            if terminal:
                if winning_move(board, AI_PIECE):
                    return (None, 1000000 + depth)
                elif winning_move(board, PLAYER_PIECE):
                    return (None, -1000000 - depth)
                else:
                    return (None, 0)  # Seri
            return (None, self.score_position(board))

        if maximizing_player:
            value = -math.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                self.last_metrics["total_branches_considered"] += 1
                row = get_next_open_row(board, col)
                b_copy = copy_board(board)
                drop_piece(b_copy, row, col, AI_PIECE)

                _, new_score = self._minimax(b_copy, depth - 1, alpha, beta, False)
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.last_metrics["pruned_branches"] += 1
                    break  # Beta cut-off
            return best_col, value

        else:
            value = math.inf
            best_col = valid_cols[0]
            for col in valid_cols:
                self.last_metrics["total_branches_considered"] += 1
                row = get_next_open_row(board, col)
                b_copy = copy_board(board)
                drop_piece(b_copy, row, col, PLAYER_PIECE)

                _, new_score = self._minimax(b_copy, depth - 1, alpha, beta, True)
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    self.last_metrics["pruned_branches"] += 1
                    break  # Alpha cut-off
            return best_col, value

    # ---------------------------------------------------------
    # 5. PIPELINE KEPUTUSAN UTAMA
    # ---------------------------------------------------------
    def choose_move(self, board):
        start_time = time.perf_counter()
        self.last_metrics = {
            "decision_source": "",
            "nodes_evaluated": 0,
            "pruned_branches": 0,
            "total_branches_considered": 0,
            "pruning_ratio": 0.0,
            "execution_time_ms": 0.0,
            "p_aggressive": round(self.p_aggressive, 3),
        }

        # Tahap 1: Cek Forward Chaining
        fc_col, source = self.forward_chaining(board)
        if fc_col is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.last_metrics["decision_source"] = source
            self.last_metrics["execution_time_ms"] = round(elapsed_ms, 3)
            return fc_col

        # Tahap 2: Jalankan Minimax + Alpha-Beta + Move Ordering
        best_col, _ = self._minimax(board, self.depth, -math.inf, math.inf, True)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        total_b = self.last_metrics["total_branches_considered"]
        pruned_b = self.last_metrics["pruned_branches"]
        ratio = (pruned_b / total_b * 100) if total_b > 0 else 0.0

        self.last_metrics["decision_source"] = "MINIMAX_ALPHA_BETA"
        self.last_metrics["pruning_ratio"] = round(ratio, 2)
        self.last_metrics["execution_time_ms"] = round(elapsed_ms, 3)

        return best_col