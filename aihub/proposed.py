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

label_map = {
    'Forward-Falls': 0,
    'Sideways-Falls': 1,
    'Backward-Falls': 2,
    'Non-Falls': 3
}

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
dataset = KeypointsSequenceDataset(data_paths, sequence_length)

train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')
test_indices = np.load('test_indices.npy')

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

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
model = TransformerModelWithAttention(num_keypoints=17, dim_model=64, num_heads=4, num_layers=3, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def train(model, train_loader, criterion, optimizer, device):
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

    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_predictions

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_attention_map(attention_map, title, path):
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(attention_map.cpu().numpy(), cmap='Reds')
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(10, 7), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(path)
    plt.close()

log_file = '/home/aibig30/data/NIA/outputs/logs/4.transformer_mul_log.txt'
with open(log_file, 'w') as f:
    f.write('Epoch,Train Loss,Train Acc,Val Loss,Val Acc,Val Precision,Val Recall,Val F1\n')

num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_labels, val_predictions = evaluate(model, val_loader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    print(f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}')
    
    scheduler.step(val_loss)

    with open(log_file, 'a') as f:
        f.write(f'{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n')

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs, attn_map_spatial, attn_map_temporal = model(sequences)
            if attn_map_spatial is not None and attn_map_temporal is not None:
                spatial_path = f'/home/aibig30/data/NIA/outputs/logs/att2/mul_spatial_attention_map_epoch_{epoch+1}.png'
                temporal_path = f'/home/aibig30/data/NIA/outputs/logs/att2/mul_temporal_attention_map_epoch_{epoch+1}.png'
                save_attention_map(attn_map_spatial[0], f'Spatial Attention Map (Epoch {epoch+1})', spatial_path)
                save_attention_map(attn_map_temporal[0], f'Temporal Attention Map (Epoch {epoch+1})', temporal_path)
            break

test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_predictions = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

conf_matrix = confusion_matrix(test_labels, test_predictions)
binary_labels = ['Forward-Falls', 'Sideways-Falls', 'Backward-Falls', 'Non-Falls']
conf_matrix_path = '/home/aibig30/data/NIA/outputs/plots/4.transformer_mul_confusion_matrix.png'
save_confusion_matrix(conf_matrix, binary_labels, conf_matrix_path)

save_model_path = '/home/aibig30/data/NIA/outputs/weights/4.transformer_mul_model.pth'
save_model(model, save_model_path)

with open('/home/aibig30/data/NIA/outputs/logs/4.transformer_mul_model_test_log.txt', 'w') as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    np.savetxt(f, conf_matrix, fmt='%d')
