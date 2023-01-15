import torch
import torch.nn as nn
import math

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor

class Representation_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()

        self.state_norm = nn.Linear(observation_space_dimensions, state_dimension)
    def forward(self, state):
        return self.state_norm(state)

class Dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension

        lstm_reward = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_state = [
            nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.reward = nn.Sequential(*tuple(lstm_reward))
        self.next_state_normalized = nn.Sequential(*tuple(lstm_state))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), self.next_state_normalized(x)


class Prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()

        lstm_policy = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions, action_dimension,number_of_hidden_layer),
            extract_tensor()
        ]

        lstm_value = [
            nn.Linear(state_dimension, hidden_layer_dimensions),
            nn.LSTM(hidden_layer_dimensions , state_dimension,number_of_hidden_layer),
            extract_tensor(),
        ]


        self.policy = nn.Sequential(*tuple(lstm_policy))
        self.value = nn.Sequential(*tuple(lstm_value))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)

