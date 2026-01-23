import torch
import torch.nn as nn

# Define MoodCNNClassifier model
class MoodCNNClassifier(nn.Module):

    def __init__(self):
        super(MoodCNNClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        # Calculate the size of the flattened layer dynamically
        # Run a dummy tensor through the conv/pool layers to find the shape
        self._to_linear = None
        dummy_tesnor_shape = (1, 91, 109, 91) # Input shape (C, D, H, W))
        self._get_conv_output(dummy_tesnor_shape)


        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 1280),
            nn.ReLU(),
            nn.Linear(1280, 256),
            nn.ReLU(),
            #nn.Dropout(0.4),   # Optional dropout layer
            nn.Linear(256, 2),  # 2 output classes
            #nn.Softmax(dim=0)  # Optional softmax layer (not needed with CrossEntropyLoss)
        )

    def _get_conv_output(self, shape):
        """
        Helper function to calculate the input size for the fully connected layers.
        
        Args:
            shape (tuple): The shape of the input tensor (C, D, H,).
        Returns:
            None: Sets the self._to_linear attribute.
        """
        # Create a dummy input tensor with batch size 1
        dummy_input = torch.rand(1, *shape) 
        output_features = self.conv_layers(dummy_input)
        # Store the calculated size
        self._to_linear = output_features.view(output_features.size(0), -1).size(1)
        #print(f"Calculated flattened feature size: {self._to_linear}") # For debugging

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x  # logits