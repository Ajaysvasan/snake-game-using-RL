import pygame
import random
import numpy as np


BLOCK_SIZE = 20
WIDTH = 400
HEIGHT = 400
SPEED = 10  


UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.block_size = BLOCK_SIZE
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game with Deep Q-Learning')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.width // BLOCK_SIZE) - 1) * BLOCK_SIZE
        y = random.randint(0, (self.height // BLOCK_SIZE) - 1) * BLOCK_SIZE
        return (x, y)

    def step(self, action):
        self._move(action)
        self.snake.insert(0, self.new_head)

        reward = 0
        if self._is_collision():
            self.game_over = True
            reward = -10
            return self.get_state(), reward, True


        if self.new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, False

    def _move(self, action):

        x, y = self.snake[0]
        if action == 1: 
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  
            self.direction = (-self.direction[1], self.direction[0])

        self.new_head = (x + self.direction[0] * BLOCK_SIZE,
                         y + self.direction[1] * BLOCK_SIZE)

    def _is_collision(self):
        x, y = self.new_head
        if (x < 0 or x >= self.width or y < 0 or y >= self.height):
            return True
        if self.new_head in self.snake[1:]:
            return True
        return False

    def get_state(self):

        state = [
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            self.snake[0][0] - self.food[0],  
            self.snake[0][1] - self.food[1]   
        ]
        return np.array(state, dtype=np.float32)

    def render(self):
        self.screen.fill((0, 0, 0)) 


        for pos in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (*pos, self.block_size, self.block_size))


        pygame.draw.rect(self.screen, (255, 0, 0), (*self.food, self.block_size, self.block_size))

        pygame.display.flip()
        self.clock.tick(SPEED)  

import tensorflow as tf
import numpy as np
import random
from collections import deque
import pygame


class DuelingDQN(tf.keras.Model):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')


        self.value_fc = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)


        self.advantage_fc = tf.keras.layers.Dense(64, activation='relu')
        self.advantage = tf.keras.layers.Dense(3) 

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        
        value = self.value_fc(x)
        value = self.value(value)

    
        advantage = self.advantage_fc(x)
        advantage = self.advantage(advantage)

    
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values

GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQUENCY = 5  

class Agent:
    def __init__(self):
        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) 
        self.epsilon = EPSILON_START
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2) 
        q_values = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
        return np.argmax(q_values.numpy()[0])

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            q_values = tf.gather_nd(q_values, np.stack((np.arange(BATCH_SIZE), actions), axis=1))

            next_q_values = self.target_model(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        
            target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values

            
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

NUM_EPISODES = 100000

agent = Agent()
game = SnakeGame()

for episode in range(NUM_EPISODES):
    state = game.reset()
    total_reward = 0
    done = False

    while not done:
        game.render()  

        action = agent.choose_action(state)
        next_state, reward, done = game.step(action)

    
        if done and reward == -10:      
            reward = -20
        elif np.linalg.norm(np.array(game.snake[0]) - np.array(game.food)) < 20:  
            reward = 10

        agent.update_memory(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        agent.update_target_model()

    agent.decay_epsilon()

    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

pygame.quit()
