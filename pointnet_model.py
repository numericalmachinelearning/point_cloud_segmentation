import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PointNet(nn.Module):
    """Simple PointNet for semantic segmentation - pure PyTorch, no compilation needed!"""
    
    def __init__(self, num_classes=37, num_features=4):
        super(PointNet, self).__init__()
        
        # Shared MLP for point features
        self.conv1 = nn.Conv1d(num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Segmentation head
        self.conv4 = nn.Conv1d(1088, 512, 1)  # 1024 global + 64 local
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (B, num_features, N) - batch of point clouds
        Returns:
            (B, num_classes, N) - per-point predictions
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Point features
        x1 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x2 = self.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = self.relu(self.bn3(self.conv3(x2)))  # (B, 1024, N)
        
        # Global feature
        global_feat = torch.max(x3, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.repeat(1, 1, num_points)  # (B, 1024, N)
        
        # Concatenate local and global features
        x = torch.cat([x1, global_feat], 1)  # (B, 1088, N)
        
        # Segmentation head
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)  # (B, num_classes, N)
        
        return x


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (points, labels) in enumerate(loader):
        points = points.to(device)  # (B, N, 4)
        labels = labels.to(device)  # (B, N)
        
        # Transpose for conv1d: (B, 4, N)
        points = points.transpose(1, 2)
        
        optimizer.zero_grad()
        outputs = model(points)  # (B, num_classes, N)
        
        # Transpose back for loss: (B, N, num_classes)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs.view(-1, outputs.size(-1))  # (B*N, num_classes)
        labels = labels.view(-1)  # (B*N)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_points = 0
    num_batches = 0
    
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            
            points = points.transpose(1, 2)
            outputs = model(points)
            outputs = outputs.transpose(1, 2).contiguous()
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            
            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()
            
            preds = outputs_flat.argmax(dim=1)
            total_correct += (preds == labels_flat).sum().item()
            total_points += labels_flat.size(0)
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * total_correct / total_points
    
    return avg_loss, accuracy


# ============================================================
# === ADDED BELOW — NOTHING ABOVE WAS TOUCHED =================
# ============================================================

import os
import pickle
from torch.utils.data import Dataset, DataLoader

class PandaSetPointCloudDataset(Dataset):
    """
    Minimal PandaSet loader for PointNet.

    Matches the structure:

        pandaset/
          001/
            annotations/
              semseg/   <-- semantic labels
            lidar/      <-- point clouds
            camera/
            meta/
    """

    def __init__(self, root, sequences=("001",), max_points=4096):
        self.items = []
        self.max_points = max_points
        self.root = root

        for seq in sequences:
            seq_dir = os.path.join(root, seq)
            lidar_dir = os.path.join(seq_dir, "lidar")
            ann_dir = os.path.join(seq_dir, "annotations")
            semseg_dir = os.path.join(ann_dir, "semseg")

            if not os.path.isdir(lidar_dir):
                print(f"[WARN] Missing lidar dir for sequence {seq}: {lidar_dir}")
                continue
            if not os.path.isdir(semseg_dir):
                print(f"[WARN] Missing semseg dir for sequence {seq}: {semseg_dir}")
                continue

            lidar_files = sorted(
                f for f in os.listdir(lidar_dir)
                if f.endswith((".pkl", ".pickle"))
            )
            label_files = sorted(
                f for f in os.listdir(semseg_dir)
                if f.endswith((".pkl", ".pickle"))
            )

            # Build a mapping from stem -> label filename for easy matching
            label_map = {}
            for lf in label_files:
                stem = os.path.splitext(lf)[0]
                label_map[stem] = lf

            for lf in lidar_files:
                stem = os.path.splitext(lf)[0]
                # try exact stem match first
                if stem in label_map:
                    label_fname = label_map[stem]
                else:
                    # fallback: first label whose stem starts with lidar stem
                    candidates = [name for name in label_files
                                  if os.path.splitext(name)[0].startswith(stem)]
                    if not candidates:
                        # no matching label found → skip this frame
                        continue
                    label_fname = candidates[0]

                self.items.append((
                    os.path.join(lidar_dir, lf),
                    os.path.join(semseg_dir, label_fname),
                ))

        print(f"[INFO] PandaSetPointCloudDataset: collected {len(self.items)} frame pairs "
              f"from sequences {list(sequences)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        lidar_path, label_path = self.items[idx]

        with open(lidar_path, "rb") as f:
            lidar = pickle.load(f)
        with open(label_path, "rb") as f:
            semseg = pickle.load(f)

        # find point array 
        if isinstance(lidar, dict):
            if "points" in lidar:
                points_np = np.asarray(lidar["points"])
            else:
                # fallback: first 2D array
                points_np = None
                for v in lidar.values():
                    arr = np.asarray(v)
                    if arr.ndim == 2:
                        points_np = arr
                        break
                if points_np is None:
                    raise KeyError(f"Cannot find point array in {lidar_path}")
        else:
            points_np = np.asarray(lidar)

        # find labels array
        if isinstance(semseg, dict):
            if "labels" in semseg:
                labels_np = np.asarray(semseg["labels"])
            elif "semseg" in semseg:
                labels_np = np.asarray(semseg["semseg"])
            else:
                labels_np = None
                for v in semseg.values():
                    arr = np.asarray(v)
                    if arr.ndim == 1 or arr.shape[0] == points_np.shape[0]:
                        labels_np = arr
                        break
                if labels_np is None:
                    raise KeyError(f"Cannot find labels array in {label_path}")
        else:
            labels_np = np.asarray(semseg)

        # Feature dimension: 6 -> crop to first 4
        if points_np.shape[1] < 4:
            raise ValueError(f"Expected at least 4 features, got {points_np.shape[1]} in {lidar_path}")
        points_np = points_np[:, :4]  # (N, 4)

        # ensure labels length matches points
        if labels_np.shape[0] != points_np.shape[0]:
            min_len = min(labels_np.shape[0], points_np.shape[0])
            points_np = points_np[:min_len]
            labels_np = labels_np[:min_len]

        points = torch.tensor(points_np, dtype=torch.float32)   # (N, 4)
        labels = torch.tensor(labels_np, dtype=torch.long)      # (N,)

        # random downsample to max_points
        N = points.shape[0]
        if N > self.max_points:
            idxs = torch.randperm(N)[:self.max_points]
            points = points[idxs]
            labels = labels[idxs]

        return points, labels


def minimal_analysis(dataset):
    pts, lbl = dataset[0]
    max_label = int(lbl.max().item())

    print("\n=== Minimal Data Analysis ===")
    print("Num points:", pts.shape[0])
    print("Feature dimension:", pts.shape[1])
    print("Example unique labels:", lbl.unique().tolist())
    print("Max label (in first sample):", max_label)

    return max_label


if __name__ == "__main__":
    print("="*60)
    print("PointNet Model Test + Dataset Loader Test")
    print("="*60)

    ROOT = "./pandaset"
    train_sequences = ["001"]
    train_ds = PandaSetPointCloudDataset(ROOT, train_sequences)

    max_label = minimal_analysis(train_ds)   # RETURNS INT
    num_classes = max_label + 1             #  42 -> 43 classes
    num_features = 4

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = PointNet(num_classes, num_features).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_epoch(model, train_loader, optimizer, criterion, epoch=1)

    print("End of training stub.")
