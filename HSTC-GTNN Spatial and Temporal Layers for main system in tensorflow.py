import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torch_geometric.nn import GCNConv, TransformerConv
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tensorflow as tf

# Define Spatial and Temporal Layers in PyTorch
class SpatialLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialLayer, self).__init__()
        self.gcn1 = GCNConv(in_channels, 128)
        self.transformer_conv = TransformerConv(128, out_channels, heads=4)

    def forward(self, x, edge_index):
        x = torch.relu(self.gcn1(x, edge_index))
        x = self.transformer_conv(x, edge_index)
        return x

class TemporalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalLayer, self).__init__()
        self.temporal_conv = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.transformer = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.temporal_conv(x))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        return x[-1]

class CrossLayerInteraction(nn.Module):
    def __init__(self, spatial_out_channels, temporal_out_channels):
        super(CrossLayerInteraction, self).__init__()
        self.fc1 = nn.Linear(spatial_out_channels + temporal_out_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, spatial_output, temporal_output):
        x = torch.cat([spatial_output, temporal_output], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)

class HSTC_GTNN(nn.Module):
    def __init__(self, node_features, temporal_seq_length, out_channels):
        super(HSTC_GTNN, self).__init__()
        self.spatial_layer = SpatialLayer(in_channels=node_features, out_channels=out_channels)
        self.temporal_layer = TemporalLayer(in_channels=node_features, out_channels=out_channels)
        self.cross_layer = CrossLayerInteraction(spatial_out_channels=out_channels, temporal_out_channels=out_channels)

    def forward(self, node_features, edge_index, temporal_seq):
        spatial_output = self.spatial_layer(node_features, edge_index)
        temporal_output = self.temporal_layer(temporal_seq)
        output = self.cross_layer(spatial_output, temporal_output)
        return output

# Use TensorFlow for Preprocessing and Logging
def preprocess_data_tf(data):
    # Example preprocessing: Normalize data with TensorFlow functions
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data = tf.keras.utils.normalize(data)
    return tf.convert_to_tensor(data, dtype=tf.float32)

# Example data preprocessing
node_features_tf = np.random.rand(10, 64)
temporal_seq_data_tf = np.random.rand(1, 10, 64)
node_features_tf = preprocess_data_tf(node_features_tf)
temporal_seq_data_tf = preprocess_data_tf(temporal_seq_data_tf)

# Convert TensorFlow tensors to PyTorch tensors for model input
node_features = torch.from_numpy(node_features_tf.numpy())
temporal_seq_data = torch.from_numpy(temporal_seq_data_tf.numpy())
edge_index = torch.randint(0, 10, (2, 20))

# Initialize Model, Optimizer, and Loss
model = HSTC_GTNN(node_features=node_features.shape[1], temporal_seq_length=temporal_seq_data.shape[1], out_channels=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# TensorBoard Logging
writer = SummaryWriter()

# Training Loop with Logging
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    output = model(node_features, edge_index, temporal_seq_data)
    target = torch.rand(1, 1)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Log to TensorBoard
    writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")

    # Additional TensorFlow-based logging
    with tf.summary.create_file_writer("logs").as_default():
        tf.summary.scalar("loss", loss.item(), step=epoch)

# Close TensorBoard writer
writer.close()
