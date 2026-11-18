import sys
import math
import pygame
from board import (
    ROWS,
    COLS,
    EMPTY,
    PLAYER_PIECE,
    AI_PIECE,
    create_board,
    copy_board,
    is_valid_location,
    get_valid_locations,
    get_next_open_row,
    drop_piece,
    winning_move,
)
from ai_agent import Connect4HybridAgent

# Warna & Konfigurasi Layar
BLUE = (24, 78, 158)
DARK_BG = (18, 22, 30)
PANEL_BG = (28, 34, 46)
RED = (231, 76, 60)
YELLOW = (241, 196, 15)
WHITE = (240, 244, 248)
MUTED = (160, 174, 192)
ACCENT = (56, 189, 248)

SQUARESIZE = 90
RADIUS = int(SQUARESIZE / 2 - 7)
BOARD_WIDTH = COLS * SQUARESIZE
BOARD_HEIGHT = (ROWS + 1) * SQUARESIZE
SIDEBAR_WIDTH = 310
WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
HEIGHT = BOARD_HEIGHT


def draw_board(screen, board, ai_agent, status_text, status_color, font, small_font):
    screen.fill(DARK_BG)

    # Gambar papan matriks 7x6
    for c in range(COLS):
        for r in range(ROWS):
            pygame.draw.rect(
                screen,
                BLUE,
                (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE),
            )
            pygame.draw.circle(
                screen,
                DARK_BG,
                (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2),
                ),
                RADIUS,
            )

    # Gambar keping yang sudah terisi
    for c in range(COLS):
        for r in range(ROWS):
            center = (
                int(c * SQUARESIZE + SQUARESIZE / 2),
                HEIGHT - int(r * SQUARESIZE + SQUARESIZE / 2),
            )
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, center, RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, center, RADIUS)

    # Panel Telemetri AI di Sisi Kanan
    panel_x = BOARD_WIDTH + 12
    pygame.draw.rect(
        screen, PANEL_BG, (panel_x, 15, SIDEBAR_WIDTH - 24, HEIGHT - 30), border_radius=12
    )

    m = ai_agent.last_metrics
    lines = [
        ("CONNECT FOUR AI (A5)", ACCENT, font),
        ("Kelompok 5 - Pengantar KA", MUTED, small_font),
        ("", WHITE, small_font),
        ("Status Permainan:", MUTED, small_font),
        (status_text, status_color, small_font),
        ("", WHITE, small_font),
        ("--- Telemetri Langkah AI ---", ACCENT, small_font),
        (f"Sumber: {m['decision_source'] or '-'}", WHITE, small_font),
        (f"Waktu Eksekusi: {m['execution_time_ms']} ms", WHITE, small_font),
        (f"Node Dievaluasi: {m['nodes_evaluated']}", WHITE, small_font),
        (f"Rasio Pruning: {m['pruning_ratio']}%", WHITE, small_font),
        ("", WHITE, small_font),
        ("--- Penalaran Bayesian ---", ACCENT, small_font),
        (f"P(Lawan Agresif): {ai_agent.p_aggressive:.2f}", WHITE, small_font),
        (f"Bobot Attack (w_atk): {ai_agent.w_attack:.2f}", WHITE, small_font),
        (f"Bobot Defense (w_def): {ai_agent.w_defense:.2f}", WHITE, small_font),
        ("", WHITE, small_font),
        ("Kontrol:", ACCENT, small_font),
        ("• Klik kolom: Jatuhkan keping", MUTED, small_font),
        ("• Tekan [R]: Ulangi permainan", MUTED, small_font),
    ]

    y_offset = 30
    for text, color, fnt in lines:
        if text:
            surf = fnt.render(text, True, color)
            screen.blit(surf, (panel_x + 14, y_offset))
        y_offset += 26

    pygame.display.update()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect Four - Hybrid AI (Kelompok 5)")

    font = pygame.font.SysFont("segoeui", 20, bold=True)
    small_font = pygame.font.SysFont("segoeui", 15)

    def reset_game():
        return create_board(), Connect4HybridAgent(depth=5), False, 0, "Giliran Anda (Merah)", RED

    board, ai_agent, game_over, turn, status_text, status_color = reset_game()
    draw_board(screen, board, ai_agent, status_text, status_color, font, small_font)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board, ai_agent, game_over, turn, status_text, status_color = reset_game()
                draw_board(screen, board, ai_agent, status_text, status_color, font, small_font)

            if event.type == pygame.MOUSEMOTION and not game_over and turn == 0:
                pygame.draw.rect(screen, DARK_BG, (0, 0, BOARD_WIDTH, SQUARESIZE))
                posx = min(event.pos[0], BOARD_WIDTH - 1)
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update((0, 0, BOARD_WIDTH, SQUARESIZE))

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and turn == 0:
                if event.pos[0] >= BOARD_WIDTH:
                    continue

                col = int(math.floor(event.pos[0] / SQUARESIZE))
                if is_valid_location(board, col):
                    prev_board = copy_board(board)
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)

                    # Perbarui keyakinan Bayesian berdasarkan langkah manusia
                    ai_agent.update_bayesian_belief(prev_board, col)

                    if winning_move(board, PLAYER_PIECE):
                        status_text = "SELAMAT! Anda Menang!"
                        status_color = RED
                        game_over = True
                    elif len(get_valid_locations(board)) == 0:
                        status_text = "Permainan Seri (Draw)!"
                        status_color = WHITE
                        game_over = True
                    else:
                        turn = 1
                        status_text = "AI sedang berpikir..."
                        status_color = YELLOW

                    draw_board(screen, board, ai_agent, status_text, status_color, font, small_font)

        # Giliran Agen Utama AI (A5)
        if turn == 1 and not game_over:
            pygame.time.wait(250)
            col = ai_agent.choose_move(board)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    status_text = "AGEN AI MENANG!"
                    status_color = YELLOW
                    game_over = True
                elif len(get_valid_locations(board)) == 0:
                    status_text = "Permainan Seri (Draw)!"
                    status_color = WHITE
                    game_over = True
                else:
                    turn = 0
                    status_text = "Giliran Anda (Merah)"
                    status_color = RED

                draw_board(screen, board, ai_agent, status_text, status_color, font, small_font)


if __name__ == "__main__":
    main()