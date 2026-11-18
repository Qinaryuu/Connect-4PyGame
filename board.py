ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2


def create_board():
    """Membuat matriks papan 7x6 (baris 0 adalah baris paling bawah)."""
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def copy_board(board):
    return [row[:] for row in board]


def is_valid_location(board, col):
    return 0 <= col < COLS and board[ROWS - 1][col] == EMPTY


def get_valid_locations(board):
    return [c for c in range(COLS) if is_valid_location(board, c)]


def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == EMPTY:
            return r
    return None


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def winning_move(board, piece):
    """Terminal / Goal Test: mengecek 4 keping sebaris (horizontal, vertikal, diagonal)."""
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == piece for i in range(4)):
                return True

    # Vertikal
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(4)):
                return True

    # Diagonal positif (/)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True

    # Diagonal negatif (\)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True

    return False


def is_terminal_node(board):
    return (
        winning_move(board, PLAYER_PIECE)
        or winning_move(board, AI_PIECE)
        or len(get_valid_locations(board)) == 0
    )