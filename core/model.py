"""
Base ML model that will receive input configurations from UI
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Map string layer types to PyTorch layers
LAYER_BUILDER = {
    "Linear": lambda params: nn.Linear(params["input_size"], params["output_size"]),
    "Conv2D": lambda params: nn.Conv2d(params["in_channels"], params["out_channels"], params["kernel_size"], params["stride"], params["padding"]),
    "MaxPooling2D": lambda params: nn.MaxPool2d(params["kernel_size"], params["stride"], params["padding"])
}

class BaseModel(nn.Module):
    def __init__(self, layer_configs):
        super().__init__()

        # Initialize layers list
        layers = []

        for config in layer_configs:            # Loop through all specified layer configurations
            layer_type = config["type"]         # Get layer type
            layer_params = config["params"]     # Get layer parameters

            # Build PyTorch layer based on type and parameters
            builder = LAYER_BUILDER[layer_type]
            layer = builder(layer_params)

            # Append layer to layers list
            layers.append(layer)

            if "activation" in layer_params:
                activation_function = self._get_activation(layer_params["activation"])

                # Append activation function to layers list
                layers.append(activation_function)

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name):
        """Get activation function from layer config"""
        activations = {
            "ReLU": nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh()
        }
        return activations.get(name, nn.ReLU())     # Get activation function, or defualt to ReLU

    def get_optimizer(self, name, learning_rate):
        """Get optimizer from layer config"""
        optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop
        }
        optimizer_class = optimizers.get(name, optim.Adam)      # Get optimizer, or default to Adam
        return optimizer_class(self.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.network(x)
