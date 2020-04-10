import torch
import torch.nn.functional as F
import numpy as np
import collections


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the
    # dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, padding=1)
        self.fc1 = torch.nn.Linear(in_features=2*2*64, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=2)

    # Function which sends some input data through the network and returns the network's output. In this example,
    # a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it
    # is just a linear layer).
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 64*2*2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# The DQN class determines how to train the above neural network.
class DQN:
    NUMBER_OF_STEPS_BEFORE_TARGET_UPDATE = 300

    # The class initialisation function.
    def __init__(self, discount=0.9, learning_rate=0.01):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network()
        # Create a target Q-network which will be used in the Bellman equation
        self.target_q_network = Network()
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # discount factor
        self.discount = discount
        # The q network stores the agent past transitions into a replay buffer
        self.replay_buffer = ReplayBuffer(max_capacity=1000000)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a
    # transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, batch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    def _calculate_loss(self, transitions):
        inputs = np.array([transitions[i, 0] for i in range(transitions.shape[0])])
        rewards = np.array([transitions[i, 2] for i in range(transitions.shape[0])])
        next_state_inputs = np.array([transitions[i, 3] for i in range(transitions.shape[0])])

        batch_inputs_tensor = torch.tensor(inputs).float()
        batch_rewards_tensor = torch.tensor(rewards).float().unsqueeze(1)
        batch_actions_tensor = torch.tensor(
            np.array([transitions[i, 1] for i in range(transitions.shape[0])])).unsqueeze(1)
        next_state_inputs_tensor = torch.tensor(next_state_inputs).float()

        # Forward inputs through networks
        next_state_target_q_values = self.target_q_network.forward(next_state_inputs_tensor).detach()
        #double_q_argmax_values = torch.tensor(np.array([t.max(0)[1].item() for t in next_state_target_q_values])).unsqueeze(1)
        #double_q_values = self.q_network.forward(next_state_inputs_tensor).detach().gather(1, double_q_argmax_values)

        # next_state_target_q_values = self.target_q_network.forward(next_state_inputs_tensor).detach().max(0)[1] DOUBLE Q
        network_q_values = self.q_network.forward(batch_inputs_tensor)

        loss = torch.nn.MSELoss()(network_q_values.gather(1, batch_actions_tensor),
                                   batch_rewards_tensor + self.discount * next_state_target_q_values.max(1)[0].unsqueeze(
                                       1))
        #loss = torch.nn.MSELoss()(network_q_values.gather(1, batch_actions_tensor),
        #                          batch_rewards_tensor + self.discount * double_q_values)

        deltas = (batch_rewards_tensor + self.discount * next_state_target_q_values.max(1)[0].unsqueeze(
            1) - network_q_values.gather(1, batch_actions_tensor)).detach().numpy()

        self.replay_buffer.update_transitions_weights(deltas)

        return loss

    # Function which copies the Q network weights into the target Q network
    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class ReplayBuffer:
    MIN_NUMBER_OF_BATCHES = 200  # Number of transitions needed before starting training

    def __init__(self, max_capacity):
        self.replay_buffer = collections.deque(maxlen=max_capacity)
        self.current_batch = None
        self.bias_weight = 0.005
        self.alpha = 2
        self.transitions_weights = []
        self.sampling_probabilities = []

    def add_sample(self, sample):
        # Add sample to buffer
        self.replay_buffer.append(sample)
        # Add weight for this new sample
        self.transitions_weights.append(0)
        # While there is no training, all transitions get the same probability of being picked up
        if len(self) == ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities = [1 / len(self)] * len(self)
        # If the network is training, then the maximum probability in the buffer is assigned to the new sample
        if len(self) > ReplayBuffer.MIN_NUMBER_OF_BATCHES:
            self.sampling_probabilities.append(max(self.sampling_probabilities))
        # The buffer has a maximum capacity of 5000 transitions. If the size reaches 5000, we remove the first 10%
        if len(self) > 5000:
            for i in range(int((len(self)/10))):
                self.replay_buffer.popleft()
                self.transitions_weights = self.transitions_weights[1:]
                self.sampling_probabilities = self.sampling_probabilities[1:]

    def get_batch(self):
        # Sort the indices based on each transition probability
        batch_indices = sorted(range(len(self.sampling_probabilities)),
                               key=lambda i: self.sampling_probabilities[i], reverse=True)[:ReplayBuffer.MIN_NUMBER_OF_BATCHES]
        # Get the selected transitions from the replay buffer
        batch = np.array([self.replay_buffer[i] for i in batch_indices])
        self.batch_indices = batch_indices
        return batch

    def __len__(self):
        return len(self.replay_buffer)

    def update_transitions_weights(self, deltas):
        batch_weights = np.abs(deltas) + self.bias_weight
        for k, i in enumerate(self.batch_indices):
            self.transitions_weights[i] = batch_weights[k].item()
        self.update_sampling_probabilities()

    def update_sampling_probabilities(self):
        sum_of_weights = sum([weight ** self.alpha for weight in self.transitions_weights])
        for i in range(len(self.sampling_probabilities)):
            self.sampling_probabilities[i] = self.transitions_weights[i] ** self.alpha / sum_of_weights

if __name__ == '__main__':
    net = Network()
    x = torch.ones(1, 4, 80, 80)
    net(x)