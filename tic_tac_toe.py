import numpy as np

class Tic_Tac_Toe:
    def __init__(self, print_board=False):
        self.board = np.zeros((3,3)) # game board
        self.print = print_board
        if self.print:
            print(self.board,'\n')
        self.winner = None
        self.turns = 0
        self.player_turns = [0,0]
    
    def get_board(self, inverted=False):
        if inverted:
            return -self.board            
        return self.board  
    
    
    def make_cross(self,position):
        board = self.board.flatten()
        if board[position]!=0:
            print('Invalid move!')
        else:
            board[position] = 1
        self.board = np.reshape(board,(3,3))
        if self.print:
            print(self.board,'\n')
        if self.has_won():
            if self.print:
                print('Victory!')
            self.winner = 0
        
    def make_circle(self,position):
        board = self.board.flatten()
        if board[position]!=0:
            print('Invalid move!')
        else:
            board[position] = -1
        self.board = np.reshape(board,(3,3))        
        if self.print:
            print(self.board,'\n')
        if self.has_won():
            if self.print:
                print('Victory!')
            self.winner = 1
        
    def has_won(self):
        # whether someone has won
        return self.horizontal_win() or self.vertical_win() or self.diagonal_win()
       
    def horizontal_win(self):
        # if someone has a horizontal winning line
        for row_idx in range(self.board.shape[0]): # iterate through rows
            if np.abs(np.sum(self.board[row_idx,:]))==3:
                return True
        return False
    
    def vertical_win(self):
        # if someone has a vertical winning line
        for column_idx in range(self.board.shape[1]): # iterate through rows
            if np.abs(np.sum(self.board[:,column_idx]))==3:
                return True
        return False
    
    def diagonal_win(self):
        # if someone has a diagonal winning line
        if np.abs(self.board[0,0] + self.board[1,1] + self.board[2,2]) == 3:
            return True
        if np.abs(self.board[0,2] + self.board[1,1] + self.board[2,0]) == 3:
            return True
        return False
    
    def possible_moves(self):
        # returns a list with all possible moves/free positions
        return np.where(self.board.flatten()==0)[0]
    
    def is_full(self):
        # whether the game board is full
        return len(self.possible_moves()) == 0
 
    def run(self,player1,player2):
        won = False # don't continue game if someone has already won
        full = len(self.possible_moves()) == 0 # if game board is full
        
        while not won and not full:
            
            # Player 1's turn:
            if self.turns%2==0:
                self.make_cross(player1.choose_action(self.get_board(),
                                                      possible_actions = self.possible_moves()))
                self.player_turns[0] += 1
            
            # Player 2's turn:
            if self.turns%2==1:
                self.make_circle(player2.choose_action(self.get_board(inverted=True),
                                                       possible_actions = self.possible_moves()))
                self.player_turns[1] += 1
            
            won = self.has_won() # Check if game is won
            full = self.is_full() # Check if game board is full
            if full and self.print:
                print('Game board is full')
            self.turns += 1

class Random_Player:   
    # player randomly chooses action from possible moves     
    def choose_action(self,game_board, possible_actions=range(9)):
        return np.random.choice(possible_actions)

class Custom_Player:   
       
    def choose_action(self,game_board, possible_actions=range(9)):
        possible_actions = np.array(possible_actions)
        choice = int(input('Please insert the field index:'))
        if np.sum(possible_actions==choice)==0: # Action is not possible
            print('Action is not possible.')
            self.choose_action(game_board, possible_actions)            
        return choice
    
class Preventive_Player():
    # Player which plays random but tries to prevent the third in a row by opponent
    def choose_action(self,game_board,possible_actions=range(9)):
        
        blocking_fields = []
        
        # is there danger?
        # vertical dangers
        vertical_dangers = np.where(np.sum(game_board,axis=0)==-2)[0]
        for column in vertical_dangers:
            blocking_fields.append(3*np.where(game_board[:,column]==0)[0] + column)
            
        # horizontal dangers
        horizontal_dangers = np.where(np.sum(game_board,axis=1)==-2)[0]
        for row in horizontal_dangers:
            blocking_fields.append(3*row + np.where(game_board[row,:]==0)[0])
            
        # diagonal dangers
        if game_board[0,0] + game_board[1,1] + game_board[2,2] == -2:
            for i in range(3):
                if game_board[i,i] == 0:
                    blocking_fields.append(3*i+i)
                    
        if game_board[2,0] + game_board[1,1] + game_board[0,2] == -2:
            for i in range(3):
                if game_board[i,2-i] == 0:
                    blocking_fields.append(3*i+2-i)
                    
        # block fields
        if len(blocking_fields)>0:
            return np.random.choice(np.array(blocking_fields).flatten())
        return np.random.choice(possible_actions)
        
        
    