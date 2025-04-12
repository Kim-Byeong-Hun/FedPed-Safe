import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import math

try:
    from thop import profile, clever_format
    USE_THOP = True
except ImportError:
    print("thop is not installed. FLOPs calculation will be skipped.")
    USE_THOP = False

import matplotlib.pyplot as plt

label_map = {
    'Falling forward using hands': 0,
    'Falling forward using knees': 1,
    'Falling backward': 2,
    'Falling sideways': 3,
    'Falling sitting in empty chair': 4,
    'Walking': 5,
    'Standing': 6,
    'Sitting': 7,
    'Picking up an object': 8,
    'Jumping': 9,
    'Laying': 10
}

class_names = list(label_map.keys())

SEED = 12340
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
                        keypoint_files = keypoint_files[:sequence_length]
                        self.data.append((keypoint_files, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoint_files, label = self.data[idx]
        sequence_data = []
        for kf in keypoint_files:
            if kf is None:
                keypoints = torch.zeros((17,3), dtype=torch.float)
            else:
                with open(kf, 'r') as f:
                    line = f.readlines()[0]
                    kp_str = line.split(', ', 1)[1]
                    kp_list = eval(kp_str)
                    keypoints = torch.tensor(kp_list, dtype=torch.float)
            sequence_data.append(keypoints)

        sequence_data = torch.stack(sequence_data)
        return sequence_data, label

data_paths = [
    '/home/aibig30/data/UP-FALL-output-processed2/Camera1',
    '/home/aibig30/data/UP-FALL-output-processed2/Camera2'
]
sequence_length = 45

dataset = KeypointsSequenceDataset(data_paths, sequence_length)
print("Total samples:", len(dataset))

label_counts = Counter(dataset.labels)
print("Label distribution in dataset:")
for lbl, cnt in label_counts.items():
    lbl_name = [k for k,v in label_map.items() if v==lbl][0]
    print(f"Label {lbl} ({lbl_name}): {cnt}")

split_indices_path = '/home/aibig30/data/nia_upfall/split_indices.npz'
split_data = np.load(split_indices_path)
train_indices = split_data['train']
val_indices   = split_data['val']
test_indices  = split_data['test']

train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)
test_dataset  = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float32) * (-math.log(10000.0) / dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]
        else:
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :, :]
        return x

class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super().__init__()
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
        super().__init__()
        self.keypoint_proj = nn.Linear(3, dim_model)
        self.pos_enc_kpt = PositionalEncoding(dim_model, batch_first=True)
        self.pos_enc_time = PositionalEncoding(dim_model, batch_first=True)

        self.spatial_encoders = nn.ModuleList([
            TransformerEncoderLayerWithAttention(dim_model, num_heads, dim_model*4)
            for _ in range(num_layers)
        ])
        self.temporal_encoders = nn.ModuleList([
            TransformerEncoderLayerWithAttention(dim_model, num_heads, dim_model*4)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(dim_model, num_classes)

        self.num_keypoints = num_keypoints
        self.dim_model = dim_model

    def forward(self, x):
        B, seq_len, num_kpt, cdim = x.size()

        x_proj = self.keypoint_proj(x)
        x_proj = x_proj.view(B*seq_len, num_kpt, self.dim_model)
        x_proj = self.pos_enc_kpt(x_proj)

        attn_map_spatial = None
        for i, layer in enumerate(self.spatial_encoders):
            x_proj, attn_weights = layer(x_proj)
            if i == len(self.spatial_encoders) - 1:
                attn_map_spatial = attn_weights

        x_proj = x_proj.view(B, seq_len, num_kpt, self.dim_model)
        x_spatial_pooled = x_proj.mean(dim=2)

        x_spatial_pooled = self.pos_enc_time(x_spatial_pooled)

        attn_map_temporal = None
        for i, layer in enumerate(self.temporal_encoders):
            x_spatial_pooled, attn_weights = layer(x_spatial_pooled)
            if i == len(self.temporal_encoders) - 1:
                attn_map_temporal = attn_weights

        x_temporal_pooled = x_spatial_pooled.mean(dim=1)
        out = self.fc(x_temporal_pooled)
        return out, attn_map_spatial, attn_map_temporal


def compute_flops_transformer(model, sequence_length=45, num_keypoints=17, device='cpu'):
    if not USE_THOP:
        return "0", "0"

    from thop import profile, clever_format
    dummy_input = torch.zeros((1, sequence_length, num_keypoints, 3), device=device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    return macs_str, params_str

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _, _ = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs, _, _ = model(sequences)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall    = recall_score(all_labels, all_preds, average='macro')
    f1        = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_preds

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(path)
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerModelWithAttention(
        num_keypoints=17,
        dim_model=64,
        num_heads=4,
        num_layers=3,
        num_classes=11
    ).to(device)

    macs_str, params_str = compute_flops_transformer(model, sequence_length=sequence_length, num_keypoints=17, device=device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    print(f"[Model Info] Params: {total_params_m:.2f}M")
    if USE_THOP:
        print(f"[FLOPs] MACs: {macs_str}, Params: {params_str}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    log_file = '/home/aibig30/data/nia_upfall/transformer_training_log_mul.txt'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write("Epoch,TrainLoss,TrainAcc,ValLoss,ValAcc,Precision,Recall,F1\n")

    num_epochs = 100
    results = []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        results.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        })
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},"
                    f"{val_loss:.4f},{val_acc:.4f},"
                    f"{val_precision:.4f},{val_recall:.4f},{val_f1:.4f}\n")

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )
    print(f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
          f"Prec: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    cm_path = '/home/aibig30/data/nia_upfall/transformer_confusion_matrix_mul.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    save_confusion_matrix(cm, class_names, cm_path)

    model_path = '/home/aibig30/data/nia_upfall/transformer_model_mul.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    with open(log_file, 'a') as f:
        f.write("\n=== Test Results ===\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n\n")

        if USE_THOP:
            f.write(f"FLOPs(MACs): {macs_str}, THOP Params: {params_str}\n")
        else:
            f.write("FLOPs: skipped (thop not installed)\n")
        f.write(f"Total trainable Params: {total_params_m:.2f} M\n")

    print("Done. Confusion matrix saved, logs updated.")
