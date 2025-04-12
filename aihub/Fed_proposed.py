import os
import glob
import torch
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

label_map = {
    'Forward-Falls': 0,
    'Sideways-Falls': 1,
    'Backward-Falls': 2,
    'Non-Falls': 3
}

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class KeypointsSequenceDataset(Dataset):
    def __init__(self, data_paths, sequence_length):
        self.data = []
        self.labels = []
        self.sequence_length = sequence_length

        for data_path in data_paths:
            for category in os.listdir(data_path):
                folder_path = os.path.join(data_path, category)
                label_name = category.split('_')[0]
                if label_name in label_map:
                    label = label_map[label_name]
                    keypoint_files = sorted(glob.glob(os.path.join(folder_path, 'keypoints_*.txt')))

                    if len(keypoint_files) > 0:
                        if len(keypoint_files) < sequence_length:
                            keypoint_files += [None] * (sequence_length - len(keypoint_files))
                        self.data.append((keypoint_files[:sequence_length], label))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoint_files, label = self.data[idx]

        sequence_data = []
        for keypoint_file in keypoint_files:
            if keypoint_file is None:
                keypoints = torch.zeros((17, 2), dtype=torch.float)
            else:
                with open(keypoint_file, 'r') as f:
                    keypoints = eval(f.readlines()[0].split(', ', 1)[1])
                    keypoints = torch.tensor(keypoints, dtype=torch.float)[:, :2]
            sequence_data.append(keypoints)

        sequence_data = torch.stack(sequence_data)

        return sequence_data, label

base_data_path = 'NIA_output_processed'
data_paths = [os.path.join(base_data_path, folder) for folder in os.listdir(base_data_path)]
sequence_length = 45
datasets = [KeypointsSequenceDataset([data_path], sequence_length) for data_path in data_paths]

train_loaders = []
val_loaders = []
test_loaders = []

for dataset in datasets:
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :, :]
        return x

class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super(TransformerEncoderLayerWithAttention, self).__init__()
        self.attention = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, attn_weights

class TransformerModelWithAttention(nn.Module):
    def __init__(self, num_keypoints, dim_model, num_heads, num_layers, num_classes):
        super(TransformerModelWithAttention, self).__init__()
        
        self.keypoint_proj = nn.Linear(2, dim_model)
        
        self.positional_encoding_keypoints = PositionalEncoding(dim_model, batch_first=True)
        self.positional_encoding_time = PositionalEncoding(dim_model, batch_first=True)

        self.spatial_encoder = nn.ModuleList([
            TransformerEncoderLayerWithAttention(dim_model, num_heads, dim_model * 4) for _ in range(num_layers)
        ])

        self.temporal_encoder = nn.ModuleList([
            TransformerEncoderLayerWithAttention(dim_model, num_heads, dim_model * 4) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(dim_model, num_classes)

    def forward(self, keypoints_seq):
        batch_size, seq_len, num_keypoints, _ = keypoints_seq.size()

        keypoints_seq = self.keypoint_proj(keypoints_seq)

        keypoints_seq = keypoints_seq.view(batch_size * seq_len, num_keypoints, self.keypoint_proj.out_features)

        keypoints_seq = self.positional_encoding_keypoints(keypoints_seq)

        attention_maps_spatial = None
        for i, layer in enumerate(self.spatial_encoder):
            keypoints_seq, attn_weights = layer(keypoints_seq)
            if i == len(self.spatial_encoder) - 1:
                attention_maps_spatial = attn_weights

        keypoints_seq = keypoints_seq.view(batch_size, seq_len, num_keypoints, self.keypoint_proj.out_features)

        keypoints_seq = keypoints_seq.mean(dim=2)

        keypoints_seq = self.positional_encoding_time(keypoints_seq)

        attention_maps_temporal = None
        for i, layer in enumerate(self.temporal_encoder):
            keypoints_seq, attn_weights = layer(keypoints_seq)
            if i == len(self.temporal_encoder) - 1:
                attention_maps_temporal = attn_weights

        pooled = keypoints_seq.mean(dim=1)

        out = self.fc(pooled)

        return out, attention_maps_spatial, attention_maps_temporal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = TransformerModelWithAttention(num_keypoints=17, dim_model=64, num_heads=4, num_layers=3, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(global_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def train_client(model, train_loader, criterion, optimizer, device, num_epochs=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs, _, _ = model(sequences)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_predictions

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(10, 7), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def federated_train(global_model, train_loaders, val_loaders, test_loaders, num_rounds, log_file, num_epochs=3, client_fraction=0.7):
    with open(log_file, 'w') as f:
        f.write('Round,Train Loss,Train Acc,Val Loss,Val Acc,Val Precision,Val Recall,Val F1\n')

    num_clients = len(train_loaders)
    all_labels = ['Forward-Falls', 'Sideways-Falls', 'Backward-Falls', 'Non-Falls']

    for round in range(num_rounds):
        num_participants = max(1, int(client_fraction * num_clients))
        selected_clients = random.sample(range(num_clients), num_participants)

        local_weights = []
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_f1s = []

        for client_idx in selected_clients:
            train_loader = train_loaders[client_idx]
            val_loader = val_loaders[client_idx]

            local_model = TransformerModelWithAttention(num_keypoints=17, dim_model=64, num_heads=4, num_layers=3, num_classes=4).to(device)
            local_model.load_state_dict(global_model.state_dict())

            optimizer = optim.Adam(local_model.parameters(), lr=0.001)

            epoch_train_losses = []
            epoch_train_accs = []
            epoch_val_losses = []
            epoch_val_accs = []
            epoch_val_precisions = []
            epoch_val_recalls = []
            epoch_val_f1s = []

            for epoch in range(num_epochs):
                train_loss, train_acc = train_client(local_model, train_loader, criterion, optimizer, device, num_epochs=1)
                epoch_train_losses.append(train_loss)
                epoch_train_accs.append(train_acc)

                val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(local_model, val_loader, criterion, device)
                epoch_val_losses.append(val_loss)
                epoch_val_accs.append(val_acc)
                epoch_val_precisions.append(val_precision)
                epoch_val_recalls.append(val_recall)
                epoch_val_f1s.append(val_f1)

            train_losses.append(np.mean(epoch_train_losses))
            train_accs.append(np.mean(epoch_train_accs))
            val_losses.append(np.mean(epoch_val_losses))
            val_accs.append(np.mean(epoch_val_accs))
            val_precisions.append(np.mean(epoch_val_precisions))
            val_recalls.append(np.mean(epoch_val_recalls))
            val_f1s.append(np.mean(epoch_val_f1s))

            local_weights.append(local_model.state_dict())

        global_state_dict = global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([local_weights[i][key] for i in range(len(local_weights))], 0).mean(0)
        global_model.load_state_dict(global_state_dict)

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        avg_val_precision = np.mean(val_precisions)
        avg_val_recall = np.mean(val_recalls)
        avg_val_f1 = np.mean(val_f1s)

        print(f'Round {round+1}/{num_rounds}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        print(f'Precision: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1 Score: {avg_val_f1:.4f}')
        
        with open(log_file, 'a') as f:
            f.write(f'{round+1},{avg_train_loss:.4f},{avg_train_acc:.4f},{avg_val_loss:.4f},{avg_val_acc:.4f},{avg_val_precision:.4f},{avg_val_recall:.4f},{avg_val_f1:.4f}\n')

        scheduler.step(avg_val_loss)

    test_losses = []
    test_accs = []
    test_precisions = []
    test_recalls = []
    test_f1s = []
    all_test_labels = []
    all_test_predictions = []

    for test_loader in test_loaders:
        local_model = TransformerModelWithAttention(num_keypoints=17, dim_model=64, num_heads=4, num_layers=3, num_classes=4).to(device)
        local_model.load_state_dict(global_model.state_dict())
        test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_predictions = evaluate(local_model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

        all_test_labels.extend(test_labels)
        all_test_predictions.extend(test_predictions)

    avg_test_loss = np.mean(test_losses)
    avg_test_acc = np.mean(test_accs)
    avg_test_precision = np.mean(test_precisions)
    avg_test_recall = np.mean(test_recalls)
    avg_test_f1 = np.mean(test_f1s)

    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}')
    print(f'Test Precision: {avg_test_precision:.4f}, Test Recall: {avg_test_recall:.4f}, Test F1 Score: {avg_test_f1:.4f}')

    if len(all_test_labels) > 0 and len(all_test_predictions) > 0:
        conf_matrix = confusion_matrix(all_test_labels, all_test_predictions, labels=list(range(len(all_labels))))
        conf_matrix_path = '/home/aibig30/data/NIA/outputs/plots/5.transformer_federated_mul_confusion_matrix2.png'
        save_confusion_matrix(conf_matrix, all_labels, conf_matrix_path)

        test_log_file = '/home/aibig30/data/NIA/outputs/logs/5.transformer_federated_mul_test_log2.txt'
        with open(test_log_file, 'w') as f:
            f.write(f"Test Loss: {avg_test_loss:.4f}\n")
            f.write(f"Test Accuracy: {avg_test_acc:.4f}\n")
            f.write(f"Test Precision: {avg_test_precision:.4f}\n")
            f.write(f"Test Recall: {avg_test_recall:.4f}\n")
            f.write(f"Test F1 Score: {avg_test_f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            np.savetxt(f, conf_matrix, fmt='%d')

    torch.save(global_model.state_dict(), '/home/aibig30/data/NIA/outputs/weights/5.transformer_federated_mul_model2.pth')

# Federated Learning 실행
num_rounds = 100
log_file = '/home/aibig30/data/NIA/outputs/logs/5.transformer_federated_mul_training_log2.txt'
federated_train(global_model, train_loaders, val_loaders, test_loaders, num_rounds, log_file, num_epochs=3, client_fraction=0.8)