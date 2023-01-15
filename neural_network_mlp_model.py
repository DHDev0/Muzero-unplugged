import torch
import torch.nn as nn
import math

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Representation_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|,, whatever you need
        linear_in = nn.Linear(observation_space_dimensions, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        self.scale = nn.Tanh()
        layernom_init = nn.BatchNorm1d(observation_space_dimensions)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        # 0.1, 0.2 , 0.25 , 0.5 parameter (first two more recommended for rl)
        dropout = nn.Dropout(0.1)
        activation = nn.ELU()   # , nn.ELU() , nn.GELU, nn.ELU() , nn.ELU

        first_layer_sequence = [
            linear_in,
            activation
        ]

        recursive_layer_sequence = [
            linear_mid,
            activation
        ]

        sequence = first_layer_sequence + \
            (recursive_layer_sequence*number_of_hidden_layer)

        self.state_norm = nn.Sequential(*tuple(sequence+[nn.Linear(hidden_layer_dimensions, state_dimension)]))  
        # self.state_norm = nn.Linear(observation_space_dimensions, state_dimension)
    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|, whatever you need
        linear_in = nn.Linear(state_dimension + action_dimension,hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_reward = nn.Linear(hidden_layer_dimensions,state_dimension)
        linear_out_state = nn.Linear(hidden_layer_dimensions, state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension + action_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.1)
        
        activation = nn.ELU()   

        first_layer_sequence = [
            linear_in,
            activation
        ]

        recursive_layer_sequence = [
            linear_mid,
            activation
        ]

        sequence = first_layer_sequence + \
            (recursive_layer_sequence*number_of_hidden_layer)

        self.reward = nn.Sequential(*tuple(sequence +[linear_out_reward]))
        self.next_state_normalized = nn.Sequential(*tuple(sequence +[linear_out_state]))

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        linear_in = nn.Linear(state_dimension, hidden_layer_dimensions)
        linear_mid = nn.Linear(hidden_layer_dimensions, hidden_layer_dimensions)
        linear_out_policy = nn.Linear(hidden_layer_dimensions,action_dimension)
        linear_out_value = nn.Linear(hidden_layer_dimensions,state_dimension)
        
        layernom_init = nn.BatchNorm1d(state_dimension)
        layernorm_recur = nn.BatchNorm1d(hidden_layer_dimensions)
        dropout = nn.Dropout(0.5)
        activation = nn.ELU()

        first_layer_sequence = [
            linear_in,
            activation
        ]

        recursive_layer_sequence = [
            linear_mid,
            activation
        ]

        sequence = first_layer_sequence + \
            (recursive_layer_sequence*number_of_hidden_layer)

        self.policy = nn.Sequential(*tuple(sequence + [linear_out_policy]))
        self.value = nn.Sequential(*tuple(sequence + [linear_out_value]))

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
# # # To improve the learning process and bound the activations,
# # # we also scale the hidden state to the same range as
# # # the action input
def scale_to_bound_action(x):
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (
    x - min_next_encoded_state
    ) / scale_next_encoded_state
    return next_encoded_state_normalized 

import numpy as np

class Loss_function:
    def __init__(self, parameter = (0), prediction = "no_transform",label = "no_transform"):
        """_
        Loss function and pre-transform.
        
        Example
        -------

        init class: 
        loss = Loss_function(prediction = "no_transform", 
                             label = "no_transform")
                             
        You could use a list of transform to apply such as ["softmax_softmax","clamp_softmax"]
        ps: if you add transform just be carefull to not add transform which break the gradient graph of pytorch
        
        Parameters
        ----------
        
        
            Transform
            ---------
            "no_transform" : return the input
            
            "softmax_transform" : softmax the input
            
            "zero_clamp_transform" : to solve log(0) 
             refer to : https://github.com/pytorch/pytorch/blob/949559552004db317bc5ca53d67f2c62a54383f5/aten/src/THNN/generic/BCECriterion.c#L27
            
            "clamp_transform" : bound value betwen 0.01 to 0.99


            Loss function
            -------------
            
            https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
            loss.kldiv
            
            https://en.wikipedia.org/wiki/Cross_entropy
            loss.cross_entropy
            
            https://en.wikipedia.org/wiki/Mean_squared_error
            loss.mse
            
            https://en.wikipedia.org/wiki/Root-mean-square_deviation
            loss.rmse
            
            https://en.wikipedia.org/wiki/Residual_sum_of_squares
            loss.square_error
            
            zero loss (set loss to 0)
            loss.zero_loss
        """
        self.transform = {
                    "no_transform" : lambda x : x ,
                    "softmax_transform_last_dim" : lambda x : torch.nn.Softmax(dim=-1)(x),
                    "softmax_transform" : lambda x : torch.nn.Softmax(dim=1)(x),
                    "clamp_transform" : lambda x : (x/10)+0.5,
                    "zero_clamp_transform" : lambda x : x + 1e-9,
                    "mean_clamp_transform" : lambda x : (x + 1e-6)/(x + 1e-6).sum(),
                    "avg_transform" : lambda x : (torch.abs(x)+ 1e-6)/(torch.abs(x)+ 1e-6).sum(),
                    "sigmoid_transform": lambda x : torch.nn.Sigmoid()(x),
                    "tanh_transform": lambda x : torch.nn.Tanh()(x),
                    "relu_transform": lambda x : torch.nn.ELU() (x),
                    "shrink_transform": lambda x : torch.nn.Softshrink(lambd=1e-3)(x),
                    "avg_expbound" : lambda x : self.avg_sum(x),
                    "scale_to_bound_action" : lambda x : self.scale_to_bound_action(x),
                    "sum" : lambda x : x.sum(dim=-1, keepdim= True)
                    }
        if isinstance(prediction,str):
            self.prediction_transform = self.transform[prediction]
        if isinstance(label,str):
            self.label_transform = self.transform[label]
            
        if isinstance(prediction,list):
            self.prediction = prediction
            self.prediction_transform = lambda x : self.multiple_transform(x,"pred")
        if isinstance(label,list):
            self.label = label
            self.label_transform = lambda x : self.multiple_transform(x,"lab")
        self.parameter = parameter
            
    def multiple_transform(self,x,dict_transform):
        if dict_transform == "pred":
            dict_transform = self.prediction
        else:
            dict_transform = self.label
        for i in dict_transform:
            x = self.transform[i](x)
        return x
    
    def kldiv(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        # print("P: ",p)
        # print("Q: ",q)
        # print("P_Q",(torch.log(p)-torch.log(q)))
        return (p*(torch.log(p)-torch.log(q))).sum(1)
    
    def cross_entropy(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return (-p*torch.log(q)).sum(1)
    
    
    def square_error(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return ((p-q)**(1/2)).sum()

    def mse(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        # print("P: ",p)
        # print("Q: ",q)
        # print("res: ",((p-q)**2).mean())
        return ((p-q)**2).mean(1)

    def rmse(self, input, target):
        p = self.label_transform(target)
        q = self.prediction_transform(input)
        return torch.sqrt(((p-q)**2).mean())
    
    def zero_loss(self, input, target):
        return(input+target).sum(1)*0
    
    def avg_sum(self,x):
        xx = x.flatten()
        cc = 0 
        for i in xx:
            if i > 1:
                v_c = i
            if i == 0:
                v_c = 1
            if  i < 1 and i > 0 :
                v_c = i
            if i < 0:
                v_c = 1/torch.abs(i)
            xx[cc] = v_c
            cc+=1
        res = xx.reshape(x.shape[:]).clone()
        return ((x*0)+res) / ((x*0)+res).sum()
      


# # # L1 Regularization
# # # Explain at : https://paperswithcode.com/method/l1-regularization
def l1(models, l1_weight_decay=0.0001):
    l1_parameters = []
    for parameter_1, parameter_2, parameter_3 in zip(models[0].parameters(), models[1].parameters(), models[2].parameters()):
        l1_parameters.extend(
            (parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l1_weight_decay * torch.abs(torch.cat(l1_parameters)).sum()


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 4]
# # # L2 Regularization manually
# # # or can be done using weight_decay from ADAM or SGD
# # # Explain at : https://paperswithcode.com/task/l2-regularization
def l2(models, l2_weight_decay=0.0001):
    l2_parameters = []
    for parameter_1, parameter_2, parameter_3 in zip(models[0].parameters(), models[1].parameters(), models[2].parameters()):
        l2_parameters.extend(
            (parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l2_weight_decay * torch.square(torch.cat(l2_parameters)).sum()

def weights_init(m):
    # # # std constant : 
    # # https://en.wikipedia.org/wiki/Fine-structure_constant
    # # https://en.wikipedia.org/wiki/Dimensionless_physical_constant
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1/137.035999) 
        torch.nn.init.normal_(m.bias, mean=0.0, std=1/137.035999) 
    if isinstance(m, nn.Conv2d):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
        torch.nn.init.normal_(m.weight, mean=0.0, std=1/137.035999) 
        torch.nn.init.normal_(m.bias, mean=0.0, std=1/137.035999) 

        
        



