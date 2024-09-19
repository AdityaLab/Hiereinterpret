import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("./dataset/D1.csv")

sequence_length = 13
# Features and target
external_features = data[
    [
        "X_256",
        "X_378",
    ]
]  # Replace with actual external feature names (Xs)
target = data["Processed History"]

# Scale the data
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))

external_features_scaled = scaler_x.fit_transform(external_features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))


# Function to create sequences of past data
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(
            np.hstack([features[i : i + seq_length], target[i : i + seq_length]])
        )  # Combine external features and past target values
        y.append(target[i + seq_length])  # Predict the next time step
    return np.array(X), np.array(y)


# Create sequences of past external features and target values
X, y = create_sequences(external_features_scaled, target_scaled, sequence_length)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train_tensors = torch.tensor(X_train, dtype=torch.float32)
X_test_tensors = torch.tensor(X_test, dtype=torch.float32)
y_train_tensors = torch.tensor(y_train, dtype=torch.float32)
y_test_tensors = torch.tensor(y_test, dtype=torch.float32)


# Define the RNN model
class CombinedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CombinedRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out


# Model parameters
input_size = X_train_tensors.shape[
    2
]  # Number of features (external + past processed history)
hidden_size = 64
output_size = 1  # Predicting the next value of Processed history
num_layers = 2
learning_rate = 0.001
num_epochs = 100

# Initialize the model
model = CombinedRNN(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train_tensors)
    loss = criterion(outputs, y_train_tensors)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model
model.eval()
predicted = model(X_test_tensors)
predicted = scaler_y.inverse_transform(predicted.detach().numpy())

# Inverse scale the true values
y_test_true = scaler_y.inverse_transform(y_test_tensors.detach().numpy())

# Display results

# plt.plot(y_test_true, label="True Processed History")
# plt.plot(predicted, label="Predicted Processed History")
# plt.legend()
# plt.show()


# Assuming RNN model and data are already loaded as shown in previous steps


def predict_rnn(input_data):
    model.eval()

    # Reshape the input data to the required format for the model (3D: [batch_size, sequence_length, input_size])
    input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(input_data_tensor)
    return predictions


# Define a function to calculate integrated gradients
def calculate_integrated_gradients(model, inputs, baseline):
    # Initialize the integrated gradients method
    ig = IntegratedGradients(model)

    # Calculate attributions (importance scores) for the inputs
    attributions, delta = ig.attribute(inputs, baseline, return_convergence_delta=True)

    return attributions, delta


# Prepare test data for explanation
i = 0  # Pick the test instance index for which we want to compute feature importances
input_data = torch.tensor(
    X_test[i].reshape(1, sequence_length, X_train.shape[2]), dtype=torch.float32
)  # 3D input
baseline_data = torch.zeros_like(input_data)  # Baseline (zero input)

# Call the Integrated Gradients method
attributions, delta = calculate_integrated_gradients(model, input_data, baseline_data)

# Convert attributions to numpy and take absolute values for magnitude
attributions_np = attributions.squeeze(0).detach().numpy()
importance_matrix = np.abs(attributions_np)

# Plot the matrix as a heatmap with blue-to-red colormap
plt.figure(figsize=(15, 6))
plt.imshow(importance_matrix.T, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Magnitude of Importance Score")
plt.xlabel("Time Step")
plt.ylabel("Features (External Variables + Processed History)")
plt.title("Magnitude of Integrated Gradients for Time-Series Prediction")

# Define y-axis labels for features (External variables + Processed history)
feature_labels = [f"External Variable {i+1}" for i in range(X_train.shape[2] - 1)] + [
    "Processed History"
]
plt.yticks(np.arange(X_train.shape[2]), feature_labels)
plt.xticks(np.arange(sequence_length), np.arange(-sequence_length, 0))

plt.show()
