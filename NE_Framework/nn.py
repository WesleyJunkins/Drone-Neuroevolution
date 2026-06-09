import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Input layer will have 4096 neurons, output layer will have 3 neurons (forward/backward, yaw_increase/yaw_decrease, shift_left/shift_right)

# Create a model class
class SimpleModel(nn.Module):
    def __init__(self, input_features, hidden_layers, output_features,
                 activation='relu', output_activation=None, dropout=0.0,
                 use_batch_norm=False, use_layer_norm=False, bias=True,
                 init_method='default'):
        """
        Create a flexible feedforward neural network.
        
        Args:
            input_features: Number of input neurons
            hidden_layers: List of integers specifying the number of neurons for each hidden layer
                           Example: [512, 256, 128] creates 3 hidden layers
            output_features: Number of output neurons
            
            activation: Activation function for hidden layers. Options:
                       'relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu', 'elu', 'swish'
            output_activation: Activation function for output layer. Options:
                              None (raw logits), 'sigmoid', 'tanh', 'softmax'
            dropout: Dropout probability (0.0 to 1.0). Applied after each hidden layer.
                    0.0 means no dropout.
            use_batch_norm: If True, adds BatchNorm1d layers after each hidden layer
            use_layer_norm: If True, adds LayerNorm layers after each hidden layer
                           (mutually exclusive with batch_norm)
            bias: If False, disables bias terms in all linear layers
            init_method: Weight initialization method. Options:
                        'default', 'xavier_uniform', 'xavier_normal', 
                        'kaiming_uniform', 'kaiming_normal', 'zeros', 'ones'
        """
        super().__init__()
        
        if use_batch_norm and use_layer_norm:
            raise ValueError("Cannot use both batch normalization and layer normalization")
        
        self.input_features = input_features
        self.hidden_layers = hidden_layers
        self.output_features = output_features
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        # Create list of layer sizes: input -> hidden layers -> output
        layer_sizes = [input_features] + hidden_layers + [output_features]
        
        # Dynamically create all layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0.0 else None
        
        for i in range(len(layer_sizes) - 1):
            # Create linear layer
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))
            
            # Add batch normalization after hidden layers (not after output)
            if use_batch_norm and i < len(layer_sizes) - 2:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # Add layer normalization after hidden layers (not after output)
            if use_layer_norm and i < len(layer_sizes) - 2:
                self.layer_norms.append(nn.LayerNorm(layer_sizes[i + 1]))
            
            # Add dropout after hidden layers (not after output)
            if dropout > 0.0 and i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout))
        
        # Initialize weights
        self._initialize_weights(init_method)
    
    def _get_activation(self, activation_name):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': lambda x: F.leaky_relu(x, negative_slope=0.01),
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'gelu': F.gelu,
            'elu': F.elu,
            'swish': lambda x: x * torch.sigmoid(x)
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}. Choose from {list(activations.keys())}")
        return activations[activation_name.lower()]
    
    def _get_output_activation(self, activation_name):
        """Get output activation function by name."""
        if activation_name is None:
            return lambda x: x
        activations = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'softmax': lambda x: F.softmax(x, dim=-1)
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown output activation: {activation_name}. Choose from {list(activations.keys())} or None")
        return activations[activation_name.lower()]
    
    def _initialize_weights(self, init_method):
        """Initialize weights using specified method."""
        for layer in self.layers:
            if init_method == 'default':
                continue  # PyTorch default initialization
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif init_method == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif init_method == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif init_method == 'zeros':
                nn.init.zeros_(layer.weight)
            elif init_method == 'ones':
                nn.init.ones_(layer.weight)
            else:
                raise ValueError(f"Unknown init_method: {init_method}")
            
            # Initialize bias if present
            if layer.bias is not None:
                if init_method in ['zeros', 'ones']:
                    nn.init.zeros_(layer.bias)
                else:
                    nn.init.zeros_(layer.bias)  # Default bias initialization
    
    def forward(self, x):
        activation_fn = self._get_activation(self.activation_name)
        output_activation_fn = self._get_output_activation(self.output_activation_name)
        
        # Process through hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            
            # Apply normalization if enabled
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            elif self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            # Apply activation
            x = activation_fn(x)
            
            # Apply dropout if enabled
            if self.dropouts is not None:
                x = self.dropouts[i](x)
        
        # Output layer
        x = self.layers[-1](x)
        x = output_activation_fn(x)
        
        return x




