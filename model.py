import torch
import torch.nn as nn
import gym
import numpy as np


class ConvNet(nn.Module):
    def __init__(self,
                 input_shape: gym.spaces,
                 action_space: int,
                 conv_filters: list,
                 fc_sizes: list,
                 initialize: bool):

        super(ConvNet, self).__init__()

        self._input = None

        layers = []

        input_size = input_shape[0] #frame stack number
        activation_func = nn.ReLU

        for out_channels, kernel, stride in conv_filters:
            conv = nn.Conv2d(input_size, out_channels, kernel_size=kernel, stride=stride)
            layers += [
                conv,
                nn.BatchNorm2d(out_channels),
                activation_func()
            ]
            input_size = out_channels

        self.conv_net = nn.Sequential(*layers)

        conv_out = int(np.prod(self.conv_net(torch.zeros(1, *input_shape)).size()))

        prev_size = conv_out

        fc_layers = []

        for i, size in enumerate(fc_sizes):
            linear = nn.Linear(prev_size, size, bias=True)
            if initialize:
                nn.init.xavier_normal_(linear.weight)
                nn.init.constant_(linear.bias, 0.0)
            fc_layers += [
                linear,
                activation_func()
            ]
            prev_size = size

        self.linear_net = nn.Sequential(*fc_layers)

        last_layer = nn.Linear(prev_size, action_space)
        if initialize:
            nn.init.xavier_normal_(last_layer.weight)
            nn.init.constant_(last_layer.bias, 0.0)

        self.last_layer = last_layer

        # Separate value layer
        value_layers = []
        input_size = input_shape[0]

        for out_channels, kernel, stride in conv_filters:
            conv = nn.Conv2d(input_size, out_channels, kernel_size=kernel, stride=stride)
            value_layers += [
                conv,
                nn.BatchNorm2d(out_channels),
                activation_func()
            ]
            input_size = out_channels

        prev_size = conv_out

        for i, size in enumerate(fc_sizes):
            linear = nn.Linear(prev_size, size, bias=True)
            if initialize:
                nn.init.xavier_normal_(linear.weight)
                nn.init.constant_(linear.bias, 0.0)
            value_layers += [
                linear,
                activation_func()
            ]
            prev_size = size

        self.hidden_value_network = nn.Sequential(*layers)

        vf_last_layer = nn.Linear(prev_size, 1)
        if initialize:
            nn.init.xavier_normal_(last_layer.weight)
            nn.init.constant_(last_layer.bias, 0.0)

        self.value_layer = vf_last_layer

    def forward(self, state):
        self._input = state
        x = self.conv_net(self._input)
        x = self.linear_net(x.view(x.size(0), -1))
        x = self.last_layer(x)
        return x

    def value_function(self):
        assert self._input is not None, "must call forward() first"
        x = self.hidden_value_network(self._input)
        value = self.value_layer(x)

        return value


class ConvNet2(nn.Module):
    def __init__(self,
                 input_shape: gym.spaces,
                 action_space: int,
                 conv_filters: list,
                 fc_sizes: list,
                 initialize: bool):

        super(ConvNet2, self).__init__()

        self._input = None
        conv = ConvBase(input_shape, action_space, conv_filters, fc_sizes, initialize)

        last_layer = nn.Linear(fc_sizes[-1], action_space)
        if initialize:
            nn.init.xavier_normal_(last_layer.weight)
            nn.init.constant_(last_layer.bias, 0.0)

        self._main_net = nn.Sequential(
            conv,
            last_layer
        )

        v_conv = ConvBase(input_shape, action_space, conv_filters, fc_sizes, initialize)

        v_last_layer = nn.Linear(fc_sizes[-1], 1)
        if initialize:
            nn.init.xavier_normal_(last_layer.weight)
            nn.init.constant_(last_layer.bias, 0.0)

        self._v_net = nn.Sequential(
            v_conv,
            v_last_layer
        )

    def forward(self, state):
        self._input = state
        x = self._main_net(self._input)
        return x

    def value_function(self):
        assert self._input is not None, "must call forward() first"
        value = self._v_net(self._input)
        return value


class ConvBase(nn.Module):
    def __init__(self,
                 input_shape: gym.spaces,
                 action_space: int,
                 conv_filters: list,
                 fc_sizes: list,
                 initialize: bool):

        super(ConvBase, self).__init__()

        self._input = None

        layers = []

        input_size = input_shape[0] #frame stack number
        activation_func = nn.ReLU

        for out_channels, kernel, stride in conv_filters:
            conv = nn.Conv2d(input_size, out_channels, kernel_size=kernel, stride=stride)
            layers += [
                conv,
                nn.BatchNorm2d(out_channels),
                activation_func()
            ]
            input_size = out_channels

        self.conv_net = nn.Sequential(*layers)

        conv_out = int(np.prod(self.conv_net(torch.zeros(1, *input_shape)).size()))

        prev_size = conv_out

        fc_layers = []

        for i, size in enumerate(fc_sizes):
            linear = nn.Linear(prev_size, size, bias=True)
            if initialize:
                nn.init.xavier_normal_(linear.weight)
                nn.init.constant_(linear.bias, 0.0)
            fc_layers += [
                linear,
                activation_func()
            ]
            prev_size = size

        self.linear_net = nn.Sequential(*fc_layers)

    def forward(self, state):
        self._input = state
        x = self.conv_net(self._input)
        x = self.linear_net(x.view(x.size(0), -1))
        return x
