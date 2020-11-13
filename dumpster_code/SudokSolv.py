import numpy as np

board = [
    [7,8,0,4,0,0,1,2,0],
    [6,0,0,0,7,5,0,0,9],
    [0,0,0,6,0,1,0,7,8],
    [0,0,7,0,4,0,2,6,0],
    [0,0,1,0,5,0,9,3,0],
    [9,0,4,0,6,0,0,0,5],
    [0,7,0,3,0,0,0,1,2],
    [1,2,0,0,0,7,4,0,0],
    [0,4,9,2,0,6,0,0,7]
]


BOARD_ZERO = np.zeros((9, 9), dtype=int)
BOARD = [
    [0,2,5,8,0,3,6,4,0],
    [0,8,6,9,2,0,1,0,0],
    [0,0,0,0,5,4,8,7,0],
    [5,6,4,0,7,1,3,0,8],
    [8,0,0,0,3,0,4,2,7],
    [2,0,0,4,8,9,0,6,0],
    [6,3,0,0,9,0,0,0,4],
    [0,0,0,1,6,8,7,3,0],
    [0,0,7,0,0,0,0,0,0]
]
BOARD_EASY =[
    [7,2,5,8,1,3,6,4,0],
    [4,8,6,9,2,7,1,5,3],
    [3,1,9,6,5,4,8,7,2],
    [5,6,4,2,7,1,3,9,8],
    [8,9,1,5,3,9,4,2,7],
    [2,7,3,4,8,9,5,6,1],
    [6,3,8,7,9,5,2,1,4],
    [9,4,2,1,6,8,7,3,5],
    [0,0,7,3,4,2,9,8,6]
]
BOARD = np.array(BOARD)
BOARD_EASY = np.array(BOARD_EASY)


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print('- - - - - - - - - - - -')

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(str(bo[i][j]) + " ")
            else:
                print(str(bo[i][j]) + " ", end="")


# return coordinates of empty slot or None is there is not, start up left
def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return i, j #rows & cols
    return None

def find_match(bo,trying = 0):
    pos = find_empty(bo)
    is_match = False
    while trying < 9:
        if is_match:
            break
        else:
            is_match = True
            trying += 1
            '''while trying in nope_list:
                trying += 1'''
            if trying > 9:
                is_match = False
                break
            if trying not in bo[pos[0]]:
                for i in range(len(board)):
                    if trying == bo[i][pos[1]]:
                        is_match = False
                if is_match:
                    if pos[0]%3 == 0:
                        if pos[1]%3 == 0:
                            if trying == board[pos[0]+1][pos[1]+1] or trying == board[pos[0]+2][pos[1]+1] or trying == board[pos[0]+1][pos[1]+2] or trying == board[pos[0]+2][pos[1]+2]:
                                is_match = False
                        elif pos[1]%3 == 1:
                            if trying == board[pos[0]+1][pos[1]+1] or trying == board[pos[0]+2][pos[1]+1] or trying == board[pos[0]+1][pos[1]-1] or trying == board[pos[0]+2][pos[1]-1]:
                                is_match = False
                        elif pos[1]%3 == 2:
                            if trying == board[pos[0]+1][pos[1]-1] or trying == board[pos[0]+2][pos[1]-1] or trying == board[pos[0]+1][pos[1]-2] or trying == board[pos[0]+2][pos[1]-2]:
                                is_match = False
                    elif pos[0]%3 == 1:
                        if pos[1]%3 == 0:
                            if trying == board[pos[0]+1][pos[1]+1] or trying == board[pos[0]-1][pos[1]+1] or trying == board[pos[0]+1][pos[1]+2] or trying == board[pos[0]-1][pos[1]+2]:
                                is_match = False
                        elif pos[1]%3 == 1:
                            if trying == board[pos[0]+1][pos[1]+1] or trying == board[pos[0]-1][pos[1]+1] or trying == board[pos[0]+1][pos[1]-1] or trying == board[pos[0]-1][pos[1]-1]:
                                is_match = False
                        elif pos[1]%3 == 2:
                            if trying == board[pos[0]+1][pos[1]-1] or trying == board[pos[0]-1][pos[1]-1] or trying == board[pos[0]+1][pos[1]-2] or trying == board[pos[0]-1][pos[1]-2]:
                                is_match = False
                    else:
                        if pos[1]%3 == 0:
                            if trying == board[pos[0]-1][pos[1]+1] or trying == board[pos[0]-2][pos[1]+1] or trying == board[pos[0]-1][pos[1]+2] or trying == board[pos[0]-2][pos[1]+2]:
                                is_match = False
                        elif pos[1]%3 == 1:
                            if trying == board[pos[0]-1][pos[1]+1] or trying == board[pos[0]-2][pos[1]+1] or trying == board[pos[0]-1][pos[1]-1] or trying == board[pos[0]-2][pos[1]-1]:
                                is_match = False
                        elif pos[1]%3 == 2:
                            if trying == board[pos[0]-1][pos[1]-1] or trying == board[pos[0]-2][pos[1]-1] or trying == board[pos[0]-1][pos[1]-2] or trying == board[pos[0]-2][pos[1]-2]:
                                is_match = False
            else:
                is_match = False

    if is_match:
        return trying
    else:
        return False


def solve_board(bo):
    play_list = []
    pos = find_empty(bo)
    while pos:
        match = find_match(bo)
        if match:
            bo[pos[0]][pos[1]] = match
        # maybe make another function just to rollback, duno




        play_list.append(pos)
        pos = find_empty(bo)
    print_board(bo)



solve_board(board)
















