import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle

def load_data(train_path):
    column_names = [
        "engine_id", "cycle", 
        "op_set_1", "op_set_2", "op_set_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]  # 21 sensors in FD001

    train_df = pd.read_csv(
        train_path, 
        sep=r'\s+',  # handles multiple spaces
        header=None,
        names=column_names
    )
    train_df = train_df.drop(columns=train_df.columns[26:])  # remove empty extra columns
    return train_df


def add_labels_train(train_df):
    train_df['RUL'] = train_df.groupby('engine_id')['cycle'].transform("max") - train_df['cycle']
    return train_df


def convert_to_binary(df, failure_threshold=30):
    df['label'] = (df['RUL'] <= failure_threshold).astype(int)
    return df


def create_sliding_windows(df, feature_cols, label_col, window_size, stride=1):
    X, y = [], []
    for engine_id, engine_df in df.groupby('engine_id'):
        engine_df = engine_df.sort_values('cycle')
        features = engine_df[feature_cols].values
        labels = engine_df[label_col].values
        for i in range(0, len(engine_df) - window_size + 1, stride):
            X.append(features[i:i + window_size])
            y.append(labels[i + window_size - 1])  # label at last timestep
    return np.array(X), np.array(y)


def create_test_windows(df, feature_cols, window_size):
    X = []
    engine_ids = []
    for engine_id, engine_df in df.groupby('engine_id'):
        engine_df = engine_df.sort_values('cycle')
        if len(engine_df) < window_size:
            continue
        window = engine_df[feature_cols].values[-window_size:]
        X.append(window)
        engine_ids.append(engine_id)
    return np.array(X), engine_ids


def create_y_test(X_test, test_engine_ids, rul_df, label_col='label'):
    y_test = []
    for engine_id in test_engine_ids:
        engine_row = rul_df[rul_df['engine_id'] == engine_id]
        if len(engine_row) == 0:
            raise ValueError(f"Engine ID {engine_id} not found in rul_df")
        last_label = engine_row[label_col].values[-1]
        y_test.append(last_label)
    return np.array(y_test)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PredictiveMaintenance(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_layers, num_classes,
                 kernel_size=3, cnn_dropout=0.2, lstm_dropout=0.3, fc_dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(cnn_dropout)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=20, device=None, verbose=True):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()
        test_loss = running_loss / total
        test_acc = correct / total

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return model


# ==================================================
# Training code only runs if executed directly
# ==================================================
if __name__ == "__main__":
    train_path = "data/train_FD001.txt"
    test_path = "data/test_FD001.txt"
    rul_path = "data/RUL_FD001.txt"

    train_df = load_data(train_path)
    train_df = add_labels_train(train_df)
    convert_to_binary(train_df)

    test_df = load_data(test_path)
    rul_df = pd.read_csv(rul_path, header=None, names=["RUL"])
    rul_df['engine_id'] = test_df['engine_id'].unique()
    convert_to_binary(rul_df)

    constant_sensors = ['sensor_1','sensor_6','sensor_5','sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    sensor_cols = [f"sensor_{i}" for i in range(1, 22) if f"sensor_{i}" not in constant_sensors]

    train_df = train_df.drop(columns=constant_sensors)
    test_df = test_df.drop(columns=constant_sensors)

    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

    WINDOW_SIZE = 20
    feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'label', 'RUL']]

    X_train, y_train = create_sliding_windows(train_df, feature_cols, 'label', WINDOW_SIZE)
    X_test, test_engine_ids = create_test_windows(test_df, feature_cols, WINDOW_SIZE)
    y_test = create_y_test(X_test, test_engine_ids, rul_df, label_col='label')

    train_dataset = SequenceDataset(X_train, y_train)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X_train.shape[2]
    cnn_channels, lstm_hidden_size, lstm_layers, num_classes = 64, 128, 1, 2
    dropout, lr, num_epochs = 0.3, 0.0001, 5

    model = PredictiveMaintenance(input_size, cnn_channels, lstm_hidden_size, lstm_layers, num_classes,
                                  cnn_dropout=dropout, lstm_dropout=dropout, fc_dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs)

    # Save model properly
    torch.save(trained_model.state_dict(), "predictive_maintenance_model.pth")
    print("Model saved successfully as predictive_maintenance_model.pth")


    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Scaler saved successfully as scaler.pkl")