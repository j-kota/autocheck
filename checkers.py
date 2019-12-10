import numpy as np
from random import * #for random number generator




"""
Conventions
         0: empty space
         1: Black Pawn
         2: White Pawn
         3: Black King
         4: White King
         5: Illegal space

We play on the black squares
"""



def make_king(board,space):
    (i,j) = space
    print(i)
    print(j)
    print( board[i,j] )

    if not (1<=board[i,j]<=4):   # if the space doesn't hold one of the 4 piece types
        raise ValueError("make_king() function was applied to invalid piece type")
        
    #lambda z: [3,4,3,4][z-1]     

    newboard = np.copy(board)

    newboard[i,j] = [3,4,3,4][board[i,j]-1]    # map 1->3, 2->4
    return newboard

def isKing(board,space):
    (i,j) = space
    return ( (board[i,j]==3) or (board[i,j]==4) )



"""
def board_from_state(state):
    
    # It does the inverse of state_from_board
    
    board = 4*state[3, :, :] + 3*state[2, :, :] + 2*state[1, :, :] + state[0, :, :]
    return board
"""



def expand_board(board):
    """
    Expands a compressed board by filling the invalid spaces
    """
    e_board = np.full((2*board[0].size, board[:, 0].size), -1)
    for i in range(e_board[:, 0].size):
        for j in range(e_board[0].size):
            if (i+j) % 2 == 0:
                e_board[i, j] = board[i, int(j/2)]

    return e_board





"""
Note - Standard orientation of the board will be from White's position
       White is Player 1
"""
def initial_board(nrows,ncols):

    if(  (not (nrows%2)==0) or (not (ncols%2==0))  ):
        raise ValueError("Board was initialized with odd number of rows or columns")
    
    board = np.zeros( (nrows,ncols), dtype=np.int16)
    
    for i in range( 0,nrows ):
        for j in range( 0,ncols ):
            if (  abs( i-j )%2 == 0 ):
                board[i,j] = 5        # mark the white squares 5 - illegal
            elif ( i <= 2 ):
                board[i,j] = 1        # Black Pawn
            elif ( nrows-i <= 3 ):
                board[i,j] = 2        # White Pawn

    return board
              




def get_state(board):
    p = lambda n, x: 1 if (x==n) else 0    # zero out all but those having value n
    q = np.vectorize(p)                    # change the function to work on arrays rather than single elements
    state = np.empty( (0,board.size), dtype=np.int16 )  # initialize the return array
    for i in range(1,4+1):
       flat = np.ndarray.flatten(  np.array( q(i,board), dtype = np.int16 )  )   # q(i,board) zeros out all but i
       state = np.append( state, flat )    # np.array() is there only to change the dtype
    return state

def get_board(state):
    # not implemented
    return 0


def switch_player(player):
    newplayer = player%2 + 1
    return newplayer

def print_board(board):
    sym = lambda i: [' ','b','w','B','W','X'][i]
    print(  np.vectorize(sym)(board) )


"""
Rules:
-There are black and white squares
-We play on the black squares
-Bottomleft and topright corner squares are black
"""



"""
Determine if a given square on the board is black
   - Doesn't depend on the board, as long as the bottom-left
       square is black

   - Checks if x,y coords have same parity, this
       definition doesn't depend on the size of the board
"""
def isBlacksquare(coords):

    (x,y) = coords
    return ( abs( x-y )%2 == 1 )




"""
Check if a space is within the bounds of the board
Return True if so, False if not
Parameter 'space' is of type (Int,Int)
"""
def bounds_check(board,space):
    return not any ( [(space[k] < 0)                      for k in [0,1] ] +
                     [(space[k] >= board.shape[(k+1)%2])  for k in [0,1] ] )
     


"""
Moves that a player's pawn can make given its position
Coordinates are based on the numpy array indexing
Returns a list of {-1,1} duples that represent moves

Note - Location of friendly and enemy piece is not considered (in this function)
       Just boundaries are considered
"""
def get_moves(board,space,player):
    
    (i,j) = space
    imax = board.shape[0]-1
    jmax = board.shape[1]-1


    if not bounds_check(board,space):
        raise ValueError("Out-of-bounds coordinates were input to the moves() function.")
    
    """ 
    DELETE WHEN READY
    if any ( [([i,j][k] < 0)                      for k in [0,1] ] +
             [([i,j][k] >= board.shape[(k+1)%2])  for k in [0,1] ] ):
        
        raise ValueError("Out-of-bounds coordinates were input to the moves() function.")
    """

    # Spaces with no legal moves
    if (board[i,j] == 5):
        raise ValueError("An invalid board piece was selected.")
           
    elif (board[i,j] == 0):
        pawnmoves_list =  []       # No piece here

    # Boundary spaces
    elif ( i == 0 ):        # top boundary
        pawnmoves_list = []       # no moves (for pawn)

    elif ( j == 0 ):        # left boundary
        pawnmoves_list = [(-1,1)]   # move right/up

    elif ( j == jmax ):     # right boundary
        pawnmoves_list = [(-1,-1)]  # move left/up

    else:
        # If none of the exceptions occurs, the chosen space/coord isn't at a boundary
        pawnmoves_list = [(-1,1),(-1,-1)] # move right/up or left/up


    # now add king moves
    if( isKing(board,(i,j)) ):
        #if at bottom, append empty
        #else if at left, append down right
        #else if at right, append down left

        if( i==imax ):
            kingmoves_list = []
        elif( j==0 ):
            kingmoves_list = [(1,1)]
        elif( j==jmax ):
            kingmoves_list = [(1,-1)]
        else:
            kingmoves_list = [(1,1),(1,-1)]

    else: # if the piece is not a king
        kingmoves_list = []

    allmoves_list = pawnmoves_list + kingmoves_list

    # filter the moves that are legal based on the current piece positions
    finalmoves_list = []
    for move in allmoves_list:
        if test_move(board,space,move,player):
            finalmoves_list.append(move)
    
    return (finalmoves_list)


#j-->
#i |
#  v     





def get_all_moves(board,player):

    moves = []
    
    for i in range(0,board.shape[0]):
        for j in range(0,board.shape[1]):
            if isFriendly(board[i][j],player):
                for move in get_moves(board,(i,j),player):
                    moves.append(  ((i,j),move)  )
    return moves
                




"""
Takes integer that represents the type of piece
Returns boolean - whether or not the piece is black (or white)
"""
def isBlackpiece(a):
    if (0 <= a <= 5):
        return ((a == 1) or (a == 3))
    else:
        raise ValueError("An invalid piece type was passed to the isBlackpiece function.")

    
def isWhitepiece(a):
    if (0 <= a <= 5):
        return ((a == 2) or (a == 4))
    else:
        raise ValueError("An invalid piece type was passed to the isWhitepiece function.")


"""
Determines whether a given piece type (integer)
is friendly or enemy, depending on the player that's asking
Recall - Player 1 is Black, Player 2 is White
"""
def isFriendly(a,player):
    if player == 1:
        return isWhitepiece(a)
    elif player == 2:
        return isBlackpiece(a)
    else:
        raise ValueError("A player value other than 1 or 2 was input to the isFriendly function.")

def isEnemy(a,player):
    if player == 1:
        return isBlackpiece(a) 
    elif player == 2:
        return isWhitepiece(a)
    else:
        raise ValueError("A player value other than 1 or 2 was input to the isFriendly function.")


    
    
"""
Decide whether a chosen move is allowed
A move is not allowed if:
    The destination is out of bounds
    The destination has a friendly piece
    The destination has an enemy piece and also any other piece behind it
    The destination has an enemy piece at the edge of the board

For reference:
         0: empty space
         1: Black Pawn
         2: White Pawn
         3: Black King
         4: White King
         5: Illegal space

"""  
def test_move(board,space,move,player):

    if not all( abs(move[i])==1 for i in [0,1] ):
        raise ValueError("An illegal move was input - moves should be made up of the elements 1 and -1")

    newspace  = (  move[0]+space[0],   move[1]+space[1])   
    (i,j) = newspace

    nextspace = (2*move[0]+space[0], 2*move[1]+space[1])   # The space behind newspace
    (m,n) = nextspace

    # check if the new space is within bounds
    if ( not (  isBlacksquare(newspace) and bounds_check(board,newspace)  )):
        return False

    # A free space is valid
    if ( board[i,j]==0 ):  
        return True

    if ( isFriendly( board[i,j],player ) ):
        return False

    # ----- If this point is reached, we know the newspace is legal and has an enemy piece
    
    # Check if the next space is within bounds
    if (  isBlacksquare(nextspace) and bounds_check(board,nextspace)  ):
        if ( board[m][n]==0 ): # If nextspace is empty
            return True
        else:
            return False   # Return False if nextspace not empty
    else:        
        return False       # Return False if nextspace out of bounds
         
    # The only conditions that return true are if newspace is empty,
    #   or if newspace has an enemy and nextspace is empty
    return False






"""
Apply a move to the board
Moves should be tested for errors in test_move()
Before they're applied with this function
"""
        #nparray,duple,duple,int <- all int 
def apply_move(board, space, move, player):     #test

    if not test_move(board, space, move, player):
        raise ValueError("Invalid move applied")

    (i,j) = space
    (di,dj) = move
    newspace = (i+di,j+dj)
    (ni,nj) = newspace
    nextspace = (i+di+di,j+dj+dj)
    (Ni,Nj) = nextspace

    newboard = np.copy(board)

    
    """
    testing
    print("board:")
    print(board)
    print_board(board)
    
    print("newboard:")
    print(newboard)
    print_board(newboard)
    """

    
    moveIsJump = False
    if isEnemy( board[ni,nj] ,player):
        moveIsJump = True

    
    if not moveIsJump:
        newboard[ni,nj] = board[i,j]
        newboard[i,j] = 0   
    
    
    else:
        # implement randomness here when ready
        newboard[ni,nj] = 0
        newboard[Ni,Nj] = board[i,j]
        board[i,j] = 0



    # make king if the boundary is reached
           

    return newboard


"""
def jumps(spaceslist, moveslist):   #test
    filter (lambda l: isEnemy) moveslist  #care
"""

    
"""
def getMoves
    if pawn get pawn moves 
    if king get pawn moves and add king moves 
"""

"""
def gameloop(nrows,ncols):

    player = 1   #used to tell whose turn it is 
    board = initial_board(nrows,ncols)
    
    # display board
    # get full collection of moves
    # do filtering (using test_move) to keep only legal moves
    # tag moves as jumps using isFriendly
    # choose a move
    # if was jump, get collection of moves again but filter only jumps
    # 
"""


    
"""
This is the Oracle
The branching factor is the length of the return list

def legal_moves(board, space, player):
    filter( lambda x: test_move(board,space,player)    # worry about filtering the nonwhites later
"""    





        
if __name__ == "__main__":



    (x,y) = (4,-1)

    print(   any( [True,False,False] )   )

    print( [i+1 for i in range (0,1+1)] + [i+2 for i in range (0,1+1)] )

    board =  initial_board( 12,12 )
    print("board:")
    print(board)
    print_board(board)
    
    imax = board.shape[0]-1
    jmax = board.shape[1]-1
    
    
    print(  get_moves(board,(2,5),1)  )
    print(  test_move( board, (imax-3,5), (1,1), 1 )   ) 
 

    newboard = apply_move(board, (imax-2,0), (-1,1), 1)
    print("newboard:")
    print(newboard)
    print_board(newboard)



    board = make_king( board,(imax,0) )
    print_board(board)

    print("bottomleft is king:")
    print(isKing(board,(imax,0)))




    testspace = (imax-2,2)

    
    (ti,tj) = testspace

    
    
    print_board(board)
    board = apply_move(board,(imax-2,2),(-1,1),1)
    print_board(board)
    print(get_moves(board,(imax-2-1,2+1),1))
    board = make_king( board, (imax-2-1,2+1) )
    print_board(board)
    
    
    
    #print("Moves for space (",ti,",",tj,"):")
    print("moves for new space:")
    print(get_moves(board,(imax-2-1,2+1),1))

    print("total moves for player 1:")
    print(get_all_moves(board,1))

    
    
    #test make_king on each piece and also isKing
  

    
    #a = state_from_board(board)
    #a = get_state(board)
    #b = expand_board(board)
    #c = board_from_state(a)
    
    #print('a',a.shape)
    #print('b',b)
    #print('c',c)






    


  

#"""
#def state_from_board(board):
#    """
#    The state_from_board takes board as an input
#        board is a 2-D numpy array where
#         0 is an empty space
#         1 is a Black Solider
#         2 is a White Solider
#         3 is a Black King
#         4 is a White King
#    The state returns a 3-D array that in which
#        [0][][]  1 if Black Solider or 0
#        [1][][]  1 if White Solider or 0
#        [2][][]  1 if Black King or 0
#        [3][][]  1 if White King or 0
#    """
#    state = np.zeros((board[0].size, board[:, 0].size, 4), dtype=np.int16)
#    state[3, :, :] = board/4
#    state[2, :, :] = board / 3 - state[3, :, :]
#    state[1, :, :] = board / 2 - state[2, :, :] - 2*state[3, :, :]
#    state[0, :, :] = board - 2*state[1, :, :] - 3*state[2, :, :] - 4*state[3, :, :]
#
#    return state
#
#def remove_piece(state,piece, x,y, ):
#    
#    """
#    :param state: the current state
#    :param piece: state for black or white, soldiers or king
#    :param x: stands for the x ccordinate
#    :param y: stand for the y coordinate
#    :return: return the new state
#    """
#    #create random number generator to decide the probability
#    #probability =
#    if(P>random):
#        state [piece][x][y] = 0
#    return state
#
#
#
#def turn_to_king(state,piece,x,y):
#
#    """
#
#    :param state:  current state
#    :param piece:  black or white soldier
#    :return: return the current state
#    :param x: x coordinate
#    :param y:  y coordinate
#
#    """
    # create random number generator to decide the probability
    # needed to be edit while probability is included
 #   if (piece == 0):
 #       state[1][x][y] = 0
 #       state[3][x][y] = 1
 #   if (piece == 1):
 #       state[2][x][y] = 0
 #       state[4][x][y] = 1
#
#    return state
#"""
