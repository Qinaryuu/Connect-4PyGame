import random
from board import (
    PLAYER_PIECE,
    AI_PIECE,
    create_board,
    copy_board,
    get_valid_locations,
    get_next_open_row,
    drop_piece,
    winning_move,
)
from ai_agent import Connect4HybridAgent
from baselines import RandomAgent, RuleBasedAgent, BFSAgent, StandardMinimaxAgent


def create_random_agent():
    return RandomAgent(PLAYER_PIECE)


def create_rule_based_agent():
    return RuleBasedAgent(PLAYER_PIECE)


def create_bfs_agent():
    return BFSAgent(max_depth=3, piece=PLAYER_PIECE)


def create_standard_minimax_agent():
    return StandardMinimaxAgent(depth=4, piece=PLAYER_PIECE)


def simulate_match(opponent_factory, hybrid_depth=4, num_games=20, random_openings=2):
    wins, losses, draws = 0, 0, 0
    a5_times, a5_nodes, a5_pruning = [], [], []
    opp_times, opp_nodes = [], []

    for g in range(num_games):
        board = create_board()
        hybrid_ai = Connect4HybridAgent(depth=hybrid_depth)
        opponent = opponent_factory()

        # Langkah pembuka acak agar variasi pertandingan objektif
        turn = g % 2  # 0 = Opponent (PLAYER_PIECE), 1 = Hybrid AI (AI_PIECE)
        for _ in range(random_openings):
            valid = get_valid_locations(board)
            if not valid:
                break
            col = random.choice(valid)
            row = get_next_open_row(board, col)
            piece = PLAYER_PIECE if turn == 0 else AI_PIECE
            drop_piece(board, row, col, piece)
            turn = 1 - turn

        while True:
            valid = get_valid_locations(board)
            if not valid:
                draws += 1
                break

            if turn == 0:
                prev_board = copy_board(board)
                col = opponent.choose_move(board)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                opp_times.append(opponent.last_metrics["execution_time_ms"])
                opp_nodes.append(opponent.last_metrics["nodes_evaluated"])

                # Update keyakinan Bayesian Agen Utama berdasarkan langkah lawan
                hybrid_ai.update_bayesian_belief(prev_board, col)

                if winning_move(board, PLAYER_PIECE):
                    losses += 1
                    break
            else:
                col = hybrid_ai.choose_move(board)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                a5_times.append(hybrid_ai.last_metrics["execution_time_ms"])
                a5_nodes.append(hybrid_ai.last_metrics["nodes_evaluated"])
                if hybrid_ai.last_metrics["decision_source"] == "MINIMAX_ALPHA_BETA":
                    a5_pruning.append(hybrid_ai.last_metrics["pruning_ratio"])

                if winning_move(board, AI_PIECE):
                    wins += 1
                    break

            turn = 1 - turn

    avg_a5_time = sum(a5_times) / len(a5_times) if a5_times else 0
    avg_a5_nodes = sum(a5_nodes) / len(a5_nodes) if a5_nodes else 0
    avg_a5_prune = sum(a5_pruning) / len(a5_pruning) if a5_pruning else 0
    avg_opp_time = sum(opp_times) / len(opp_times) if opp_times else 0
    avg_opp_nodes = sum(opp_nodes) / len(opp_nodes) if opp_nodes else 0

    return {
        "opponent": opponent_factory().name,
        "games": num_games,
        "win_rate": round((wins / num_games) * 100, 1),
        "record": f"{wins}W-{losses}L-{draws}D",
        "a5_time_ms": round(avg_a5_time, 2),
        "a5_nodes": round(avg_a5_nodes, 1),
        "a5_pruning_pct": round(avg_a5_prune, 1),
        "opp_time_ms": round(avg_opp_time, 2),
        "opp_nodes": round(avg_opp_nodes, 1),
    }


if __name__ == "__main__":
    print("=== BENCHMARK PERFORMA AGEN UTAMA (A5) CONNECT FOUR ===\n")
    opponents = [
        create_random_agent,
        create_rule_based_agent,
        create_bfs_agent,
        create_standard_minimax_agent,
    ]

    header = f"{'Lawan Pembanding':<24} | {'Win Rate':<8} | {'W-L-D':<9} | {'A5 Time':<9} | {'A5 Nodes':<9} | {'Pruning':<8} | {'Opp Time':<9} | {'Opp Nodes'}"
    print(header)
    print("-" * len(header))

    for factory in opponents:
        n_games = 10 if "Minimax" in factory().name else 25
        res = simulate_match(factory, hybrid_depth=4, num_games=n_games)
        print(
            f"{res['opponent']:<24} | "
            f"{res['win_rate']:>6.1f}% | "
            f"{res['record']:<9} | "
            f"{res['a5_time_ms']:>6.2f} ms | "
            f"{res['a5_nodes']:>9.1f} | "
            f"{res['a5_pruning_pct']:>6.1f}% | "
            f"{res['opp_time_ms']:>6.2f} ms | "
            f"{res['opp_nodes']:>9.1f}"
        )