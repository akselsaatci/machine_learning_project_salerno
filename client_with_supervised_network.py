from client import Client, JOINTS, DEFAULT_PORT
import sys
import math
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch.nn as nn

import pickle
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
# Load and process the CSV file
file_path = './data_new.csv'
df = pd.read_csv(file_path)

# Extract the input and output data
X = df[['bx', 'by', 'bz']].values
result_column = df['result']

# Convert the 'result' column from string to a list
y = np.array([ast.literal_eval(result_str) for result_str in result_column])

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
print(X_train)
class BallToJointsNN(nn.Module):
    def __init__(self, input_size=3, output_size=11):  # 11 joints * 3 coordinates each
        super(BallToJointsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = BallToJointsNN()



# Load the model from a pickle file
with open("./ball_to_joints_model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)





def run(cli):
    while True:
        state = cli.get_state()
    # Example new ball position
        new_ball_position = np.array([[state[17], state[18], state[19]]])
        new_ball_position_scaled = scaler_X.transform(new_ball_position)
        new_ball_position_tensor = torch.tensor(new_ball_position_scaled, dtype=torch.float32)
        
        loaded_model.eval()
        with torch.no_grad():
            predicted_joint_positions_scaled = loaded_model(new_ball_position_tensor).numpy()
            predicted_joint_positions = scaler_y.inverse_transform(predicted_joint_positions_scaled)
        
        # Reshape the output into a 2D array
        num_joints = 11
        reshaped_output = predicted_joint_positions.reshape(num_joints) 
        print(reshaped_output)
        print("running!!")
        cli.send_joints(reshaped_output)

def main():
    name = 'Example Client'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)

if __name__ == '__main__':
    main()
