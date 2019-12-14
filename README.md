# autocheck
AI for Checkers
# Game Rules
The game follows most os the rule of checkers with two majority of changes: 1. When jumping accross an enemy, the enemy will be remove base on a probability, so it is not gurantee to be removed. 2. When a pawn(soldier) reach the last row of the opponents territory, it holds a probability of turning into a king, so it is not gurantee to turn into a king as well.
# How to run
# Game Functions
```bash
def make_king(board,space)
def isKing(board,space)
def expand_board(board)
def initial_board(nrows,ncols)
def compress_board(board)
def get_state(board, player)
def get_state(board)
def get_board(state)
def switch_player(player)
def print_board(board)
def isBlacksquare(coords)
def bounds_check(board,space)
def get_moves(board,space,player)
def get_all_moves(board,player)
def get_jumps(board,space,player)
def isBlackpiece(a)
def isWhitepiece(a)
def isFriendly(a,player)
def isEnemy(a,player)
def test_move(board,space,move,player)
def apply_move(board, space, move, player)
def isTerminal(board)
def gameloop(nrows,ncols)

```

