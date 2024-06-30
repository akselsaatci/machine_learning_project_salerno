import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


inputFile = "data.csv"

data = np.genfromtxt(inputFile, delimiter=',', skip_header=1)

# Split data into input and output
input_data = data[:, :3]  # Ball positions [x, y, z]
output_data = data[:, 3:]  # Joint positions



# Generate synthetic data for demonstration purposes
# In practice, you would load your dataset
num_samples = 1000
num_joints = 11
input_data = np.random.rand(num_samples, 3)  # Ball positions [x, y, z]
output_data = np.random.rand(num_samples, num_joints * 3)  # Joint positions

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(input_data, output_data, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)



class BallToJointsNN(nn.Module):
    def __init__(self, input_size=3, output_size=num_joints * 3):
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


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 32

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test).item()
    test_mae = torch.mean(torch.abs(test_outputs - y_test)).item()

print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')


# Predict joint positions for a new ball position
new_ball_position = np.array([[0.5, 0.5, 0.5]])
new_ball_position_scaled = scaler_X.transform(new_ball_position)
new_ball_position_tensor = torch.tensor(new_ball_position_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_joint_positions_scaled = model(new_ball_position_tensor).numpy()
    predicted_joint_positions = scaler_y.inverse_transform(predicted_joint_positions_scaled)

print('Predicted Joint Positions:', predicted_joint_positions)
