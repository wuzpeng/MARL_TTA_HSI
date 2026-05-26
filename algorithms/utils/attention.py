import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import ast
import threading
import keyboard


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)

        # self.im = None
        # self.fig, self.ax = plt.subplots(figsize=(6, 1))
        # plt.ion()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.unsqueeze(0)

        # Self-attention calculation
        attention_output, attention_weights = self.multihead_attn(x, x, x)
        attention_output = attention_output.squeeze(0)

        return attention_output

    def visualize_fused_weights(self, fused_weights):
        """ Visualize fused_weights using a heatmap in real-time. """
        # Convert the fused_weights tensor to a numpy array for visualization
        fused_weights_np = fused_weights.cpu().numpy()

        if self.im is None:
            self.im = self.ax.imshow(fused_weights_np, cmap='viridis', aspect='auto')
            self.fig.colorbar(self.im)
        else:
            # Update the heatmap with the new data
            self.im.set_data(fused_weights_np)

        # Pause to update the figure in real-time
        plt.pause(0.001)  # Adjust the pause time to control the update rate


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.input_data = None
        self.fused_weights = None

        # self.im = None
        # self.fig, self.ax = plt.subplots(figsize=(6, 1))
        # plt.ion()

    def forward(self, query, keys, values):
        # Multi-head attention calculation
        if isinstance(query, np.ndarray):
            query = torch.tensor(query, dtype=torch.float32)
        if isinstance(keys, np.ndarray):
            keys = torch.tensor(keys, dtype=torch.float32)
        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=torch.float32)
        query = query.unsqueeze(0)  # 在第一个维度插入 batch_size
        keys = keys.unsqueeze(0)
        values = values.unsqueeze(0)

        attention_output, attention_weights = self.multihead_attn(query, keys, values)
        # attention_weights = attention_weights.squeeze(0)

        # Start a thread to check for user input (space key)
        # input_thread = threading.Thread(target=self.check_space_key)
        # input_thread.start()
        # input_thread.join()  # Wait for the thread to finish
        #
        # if self.input_data:
        #     user_input = self.input_data
        #     try:
        #         user_input_data = ast.literal_eval(user_input)  # Convert input string to a list
        #         user_input_tensor = torch.tensor(user_input_data, dtype=attention_weights.dtype)
        #         user_input_tensor = user_input_tensor.to(attention_weights.device)
        #
        #         # Check if the input shape matches the attention weights shape
        #         if user_input_tensor.shape == attention_weights.shape:
        #             self.fused_weights = attention_weights + user_input_tensor  # Fuse weights
        #         else:
        #             print("Input data shape does not match attention weights shape. Skipping addition.")
        #     except Exception as e:
        #         print(f"Invalid input format. Skipping addition. Error: {e}")
        # else:
        #     self.fused_weights = attention_weights

        self.fused_weights = attention_weights
        # self.visualize_fused_weights(self.fused_weights)

        # Combine fused weights with values
        attention_output = torch.matmul(self.fused_weights, values)

        return attention_output, self.fused_weights

    def check_space_key(self):
        if keyboard.is_pressed('space'):
            input_str = input("Please input data matching the shape of attention weights: ")
            self.input_data = input_str  # Update input

    def visualize_fused_weights(self, fused_weights):
        """ Visualize fused_weights using a heatmap in real-time. """
        # Convert the fused_weights tensor to a numpy array for visualization

        fused_weights_np = fused_weights.cpu().numpy()
        vmin = fused_weights_np.min()
        vmax = fused_weights_np.max()

        if self.im is None:
            self.im = self.ax.imshow(fused_weights_np, cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
            self.fig.colorbar(self.im)
        else:
            # Update the heatmap with the new data and dynamic range
            self.im.set_data(fused_weights_np)
            self.im.set_clim(vmin, vmax)  # Update color limits dynamically

        # Pause to update the figure in real-time
        plt.pause(0.001)  # Adjust the pause time to control the update rate
