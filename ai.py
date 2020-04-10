import torch
import numpy as np
from dqn import DQN, ReplayBuffer
from game import Game
import time
import cv2
import pygame


class AI:
    # Function to initialise the agent
    def __init__(self):
        # The deep q network will be trained on agent's experience
        self.dqn = DQN()
        # State 4*80*80
        self.state = None
        self.action = 0
        self.num_steps_taken = 0
        # The agent will use epsilon-greedy policy as a trade-off between exploration and exploitation
        self.epsilon = 0.25
        self.decay = 0.0003

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Compute the Q-values for this state
        q_values_tensor = self.dqn.q_network.forward(torch.tensor([state]).float())
        # Determine the action that leads to the highest Q-value in this state
        greedy_action = q_values_tensor.max(1)[1].item()
        action = np.random.choice(np.arange(2), 1, p=np.array(
            [(1 - self.epsilon) if i == greedy_action else self.epsilon for i in range(2)]))[0]
        print(f' Action: {action} - Greedy: {greedy_action} - Epsilon: {self.epsilon}')
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_reward(self, next_state, reward):
        self.num_steps_taken += 1
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add transition to the agent's replay buffer
        self.dqn.replay_buffer.add_sample(transition)
        # If at least one batch can be extracted from the buffer, train the network on a random batch
        if len(self.dqn.replay_buffer) >= ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            mini_batch = self.dqn.replay_buffer.get_batch()
            loss = self.dqn.train_q_network(mini_batch)
        if self.num_steps_taken % DQN.NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE == 0:
            self.dqn.update_target_q_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        q_values_tensor = self.dqn.q_network.forward(torch.tensor(state).float())
        action = q_values_tensor.max(1)[1].item()
        return action


if __name__ == '__main__':
    agent = AI()
    game = Game()

    start_time = time.time()
    end_time = start_time + 600  #10min

    state = game.state

    while time.time() < end_time:
        resized_state = cv2.resize(state, (80, 80), interpolation=cv2.INTER_AREA)
        stacked_state = np.stack((resized_state, resized_state, resized_state, resized_state), axis=0)
        # Get the state and action from the agent
        action = agent.get_next_action(stacked_state)
        # Get the next state and reward
        game.frame_step(action)
        next_state, reward = game.state, game.reward
        # Return this to the agent
        resized_next_state = cv2.resize(next_state, (80, 80), interpolation=cv2.INTER_AREA)
        stacked_next_state = np.stack((resized_next_state, stacked_state[0], stacked_state[1], stacked_state[2]), axis=0)
        agent.set_next_state_and_reward(stacked_next_state, reward)

        state = next_state

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()

    print('Training done!')