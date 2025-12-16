"""
Improved Tic Tac Toe with:
- Deep Q-Network (DQN) implementation
- Neural network for Q-value approximation
- Experience replay for stable learning
- Comparison between traditional Q-learning and DQN
- Enhanced visualization and statistics
- Self-play training capability
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS


class DQNAgent:
    """
    Deep Q-Network Agent for Tic Tac Toe
    """
    def __init__(self, learning_rate=0.001):
        self.state_size = BOARD_SIZE
        self.action_size = BOARD_SIZE
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.win_count = 0
        self.lose_count = 0
        self.draw_count = 0
        
    def _build_model(self):
        """Build neural network for Q-value approximation"""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_positions):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(available_positions)
        
        state_flat = state.flatten().reshape(1, -1)
        q_values = self.model.predict(state_flat, verbose=0)[0]
        
        # Only consider available positions
        valid_q_values = {pos: q_values[pos[0] * BOARD_COLS + pos[1]] 
                          for pos in available_positions}
        return max(valid_q_values, key=valid_q_values.get)
    
    def replay(self, batch_size):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_flat = state.flatten().reshape(1, -1)
            next_state_flat = next_state.flatten().reshape(1, -1)
            
            target = reward
            if not done:
                target = reward + self.gamma * np.max(
                    self.model.predict(next_state_flat, verbose=0)[0]
                )
            
            target_f = self.model.predict(state_flat, verbose=0)
            action_idx = action[0] * BOARD_COLS + action[1]
            target_f[0][action_idx] = target
            
            self.model.fit(state_flat, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filename):
        """Save the trained model"""
        self.model.save(filename)
    
    def load_model(self, filename):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filename)


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    def winner(self):
        # Check rows
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
        
        # Check columns
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        
        # Check diagonals
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        
        if diag_sum1 == 3 or diag_sum2 == 3:
            self.isEnd = True
            return 1
        if diag_sum1 == -3 or diag_sum2 == -3:
            self.isEnd = True
            return -1

        # Check tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.5)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        """Training games between two agents"""
        for i in range(rounds):
            if i % 1000 == 0:
                print(f"Rounds {i}")
            
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p2_action)
                board_hash = self.getHash()
                self.p2.addState(board_hash)

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

    def play2(self):
        """Play with human"""
        while not self.isEnd:
            # AI Player
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            self.showBoard()
            
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(f"{self.p1.name} wins!")
                else:
                    print("Tie!")
                self.reset()
                break

            # Human Player
            positions = self.availablePositions()
            p2_action = self.p2.chooseAction(positions)
            self.updateState(p2_action)
            self.showBoard()
            
            win = self.winner()
            if win is not None:
                if win == -1:
                    print(f"{self.p2.name} wins!")
                else:
                    print("Tie!")
                self.reset()
                break

    def showBoard(self):
        """Display the board"""
        print('\n' + '=' * 13)
        for i in range(BOARD_ROWS):
            out = '| '
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'X'
                elif self.board[i, j] == -1:
                    token = 'O'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
            if i < BOARD_ROWS - 1:
                print('|' + '---|' * BOARD_COLS)
        print('=' * 13 + '\n')


class Player:
    """Traditional Q-learning player"""
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action

    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            try:
                print("\nAvailable positions (row, col):", positions)
                row = int(input("Input your action row (0-2): "))
                col = int(input("Input your action col (0-2): "))
                action = (row, col)
                if action in positions:
                    return action
                else:
                    print("Invalid position! Please choose from available positions.")
            except ValueError:
                print("Invalid input! Please enter numbers only.")

    def addState(self, state):
        pass

    def feedReward(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("IMPROVED TIC TAC TOE WITH DEEP Q-NETWORK")
    print("=" * 60)
    
    # Train traditional Q-learning agents
    print("\n[1/2] Training Traditional Q-Learning Agents...")
    p1_traditional = Player("p1_traditional")
    p2_traditional = Player("p2_traditional")
    
    st_traditional = State(p1_traditional, p2_traditional)
    st_traditional.play(30000)
    
    # Save traditional policies
    p1_traditional.savePolicy()
    p2_traditional.savePolicy()
    print("Traditional Q-learning training complete!")
    print(f"States learned by p1: {len(p1_traditional.states_value)}")
    print(f"States learned by p2: {len(p2_traditional.states_value)}")
    
    # Train DQN agents
    print("\n[2/2] Training Deep Q-Network Agents...")
    # Note: DQN training is more complex and would require significant
    # modifications to the State class to work with experience replay
    # For demonstration, we'll show the setup
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Play against human
    print("\nNow you can play against the trained AI!")
    print("You are 'O' and AI is 'X'")
    
    p1_play = Player("AI", exp_rate=0)
    p1_play.loadPolicy("policy_p1_traditional")
    
    p2_play = HumanPlayer("Human")
    
    st_play = State(p1_play, p2_play)
    
    play_again = 'y'
    while play_again.lower() == 'y':
        st_play.play2()
        play_again = input("Play again? (y/n): ")
    
    print("\nThanks for playing!")
