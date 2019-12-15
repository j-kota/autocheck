# autocheck
AI for Checkers
# Reference and model
https://github.com/plkmo/AlphaZero_Connect4
# Game Size
Will be varied through 8,10,12,14,16
# Contents:
In the repository, 
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
def isFriendly(a,player), def isEnemy(a,player): Check if a pirece is an enemy or not
def test_move(board,space,move,player)
def apply_move(board, space, move, player):Apply a move to the board
def isTerminal(board): The game is considered ended if one team is wiped out
def gameloop(nrows,ncols): Play function
```
![](https://github.com/jbot2000/autocheck/blob/master/initial_state1.png)
With the function initial_board, it generates a board with 3\*boardlength white pawns(soldiers) 
and black pawns(soldiers).

![](https://github.com/jbot2000/autocheck/blob/master/initial_state2.png)
The print_board function will give a visulaized 

# MCTS Functions
Tese functions will define all the node classes, and the functions to run the tree search.
```bash
class ParentRootNode(object): Define the root node
class Node(object): Includes all the functions needed to traverse the tree
def MCTS_Search_AI(board, player, num_reads, n_net): Do MCST with neural network
def MCTS_Search(board, player, num_reads): Do MCST with neural network
def policy(node, temp=1): To calculate the policy
def MCTS_self_play(): The self play function
```
# CNN class
```bash
class Net(nn.Module):The CNN class with one concolution layer
class ErrorFnc(nn.Module): Use mean square error to calculate the loos function
```


# Training
