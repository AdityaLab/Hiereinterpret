import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lime.lime_tabular import LimeTabularExplainer
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

    # Reshape the LIME input (2D) back to 3D (batch_size, sequence_length, input_size)
    batch_size = input_data.shape[0]
    input_data_reshaped = input_data.reshape(
        batch_size, sequence_length, -1
    )  # Reshape back to 3D

    input_data_tensor = torch.tensor(input_data_reshaped, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(input_data_tensor)

    return predictions.detach().numpy()


X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Initialize the LIME explainer
explainer = LimeTabularExplainer(
    X_train.reshape(X_train.shape[0], -1),  # Flatten the input sequence to 2D
    mode="regression",  # We are doing regression
    training_labels=y_train,  # Training labels (Processed history)
    feature_names=[
        f"time_{t}_feature_{f}"
        for t in range(sequence_length)
        for f in range(X_train.shape[2])
    ],
    verbose=True,
    discretize_continuous=True,
)

# Explain a specific test instance (e.g., the first one)
i = 0  # You can change this index to test other instances
explanation = explainer.explain_instance(
    X_test_2D[i],  # Input data (1D flattened time-series sequence)
    predict_rnn,  # Prediction function
    num_features=X_train.shape[1],  # Extract all features for visualization
)

# Get the importance scores for all features at each time-step
importance_scores = dict(
    explanation.as_map()[1]
)  # Get the explanation for label 1 (regression)
print("Feature importances for the first test instance:")
print(importance_scores)

# Prepare the matrix
num_features = X_train.shape[
    2
]  # Number of external variables + 1 for Processed history
importance_matrix = np.zeros((num_features, sequence_length))

# Fill the matrix with the importance scores
for feature_index, importance in importance_scores.items():
    # The feature_index encodes both the time step and the feature index
    time_step = feature_index // num_features  # Calculate the time step
    feature = feature_index % num_features  # Calculate the feature index
    importance_matrix[feature, time_step] = abs(importance)

# Plot the matrix as a heatmap
plt.figure(figsize=(15, 6))
plt.imshow(importance_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Importance Score")
plt.xlabel("Time Step")
plt.ylabel("Features (External Variables + Processed History)")
plt.title("Importance Scores for Time-Series Prediction")

# Define y-axis labels for features (External variables + Processed history)
feature_labels = [f"External Variable {i+1}" for i in range(num_features - 1)] + [
    "Processed History"
]
plt.yticks(np.arange(num_features), feature_labels)
plt.xticks(np.arange(sequence_length), np.arange(-sequence_length, 0))
plt.show()


hist_importance = importance_matrix[-1, :]  # Importance of the processed history

print("Importance scores for the processed history:")
print(hist_importance)

avg_cumulative_importance = np.zeros(sequence_length)
for i in range(sequence_length):
    avg_cumulative_importance[i] = np.mean(hist_importance[i:])

# Plot the average cumulative importance scores
plt.figure(figsize=(10, 4))
plt.plot(np.arange(-sequence_length, 0), avg_cumulative_importance, marker="o")
plt.xlabel("Time Step")
plt.ylabel("Average Cumulative Importance Score")
plt.title("Average Cumulative Importance Scores for Processed History")
plt.grid(True)

plt.show()
