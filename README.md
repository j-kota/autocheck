# autocheck
AI for Checkers
# Game Rules
The game follows most os the rule of checkers with two majority of changes: 1. When jumping accross an enemy, the enemy will be remove base on a probability, so it is not guarantee to be removed. 2. When a pawn(soldier) reach the last row of the opponents territory, it holds a probability of turning into a king, so it is not guarantee to turn into a king as well.
# How to run
# Game Functions
```bash
def make_king(board,space): Switch a pawn(soldier) to a king
def isKing(board,space): Check if a piece is king
def expand_board(board):
def initial_board(nrows,ncols): Creates the initial state of the board, white is 1, black is 2, 
0 stands for the position a piece can be move to, 5 stands for the place where no pieces can be move to
def compress_board(board)
def get_state(board, player)
def get_state(board)
def get_board(state)
def switch_player(player): To switch player between white and black
def print_board(board): Represent the board in characters, W: white, B:black, 
x: position where no piece can be move to
def isBlacksquare(coords): 
def bounds_check(board,space)
def get_moves(board,space,player)
def get_all_moves(board,player)
def get_jumps(board,space,player)
def isBlackpiece(a), def isWhitepiece(a): Takes integer that represents the type of piece
Returns boolean - whether or not the piece is black (or white)
def isFriendly(a,player)
def isEnemy(a,player)
def test_move(board,space,move,player)
def apply_move(board, space, move, player)
def isTerminal(board)
def gameloop(nrows,ncols)

```

