import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.makedirs("models", exist_ok=True)

# 1. Dataset Windowing
class GlucoTwinDataset(Dataset):
    def __init__(self, ehr_file="data/synthetic_ehr.csv", wearable_file="data/synthetic_wearable.csv", 
                 is_train=True, split=0.8):
        df_ehr = pd.read_csv(ehr_file)
        df_wearable = pd.read_csv(wearable_file)

        patients = df_ehr["patient_id"].unique()
        split_point = int(len(patients) * split)
        selected_patients = patients[:split_point] if is_train else patients[split_point:]

        df_ehr = df_ehr[df_ehr["patient_id"].isin(selected_patients)]
        df_wearable = df_wearable[df_wearable["patient_id"].isin(selected_patients)]

        # Normalize features
        self.ehr_cols = ["age", "bmi", "baseline_hba1c", "med_metformin", "insulin_dependent"]
        ehr_vals = df_ehr[self.ehr_cols].values
        self.ehr_mean, self.ehr_std = ehr_vals.mean(axis=0), ehr_vals.std(axis=0) + 1e-7
        df_ehr[self.ehr_cols] = (ehr_vals - self.ehr_mean) / self.ehr_std

        self.dyn_cols = ["cgm_glucose", "hrv", "steps"]
        dyn_vals = df_wearable[self.dyn_cols].values
        self.dyn_mean, self.dyn_std = dyn_vals.mean(axis=0), dyn_vals.std(axis=0) + 1e-7
        df_wearable[self.dyn_cols] = (dyn_vals - self.dyn_mean) / self.dyn_std

        self.samples = []
        seq_len = 24    # 2 hours of history (24 * 5 mins)
        horizon = 24    # 2 hours into the future

        for pid in selected_patients:
            static_vec = df_ehr[df_ehr["patient_id"] == pid][self.ehr_cols].values[0]
            p_wear = df_wearable[df_wearable["patient_id"] == pid].sort_values("step_index")
            
            raw_cgm = p_wear["cgm_glucose"].values
            features = p_wear[self.dyn_cols].values
            
            # Slice into rolling 2-hour windows with 30-min stride
            for i in range(0, len(features) - seq_len - horizon, 6):
                x_dyn = features[i : i + seq_len]
                # Label is 1 if normalized future glucose exceeds ~180 mg/dL threshold
                future_val = raw_cgm[i + seq_len + horizon - 1]
                label = 1.0 if future_val > 1.1 else 0.0
                
                self.samples.append((
                    torch.tensor(static_vec, dtype=torch.float32),
                    torch.tensor(x_dyn, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 2. Multimodal Network Architecture
class GlucoTwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Dynamic Time-Series Branch (LSTM)
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        # Static Clinical Branch (Dense)
        self.ehr_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        # Multimodal Fusion Head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, static_x, dynamic_seq):
        _, (hn, _) = self.lstm(dynamic_seq)
        dynamic_feat = hn[-1]
        static_feat = self.ehr_fc(static_x)
        combined = torch.cat((dynamic_feat, static_feat), dim=1)
        return self.classifier(combined).squeeze(-1)

# 3. Training Routine
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    train_data = GlucoTwinDataset(is_train=True)
    val_data = GlucoTwinDataset(is_train=False)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    model = GlucoTwinModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for s_x, d_seq, y in train_loader:
            s_x, d_seq, y = s_x.to(device), d_seq.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(s_x, d_seq)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            
        # Validation
        model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for s_x, d_seq, y in val_loader:
                s_x, d_seq, y = s_x.to(device), d_seq.to(device), y.to(device)
                pred = model(s_x, d_seq)
                bin_pred = (pred >= 0.5).float()
                correct += (bin_pred == y).sum().item()
                count += len(y)
                
        val_acc = correct / count
        print(f"Epoch {epoch:02d}/10 | Train Loss: {total_loss/len(train_data):.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/glucotwin_model.pt")

    print(f"\n Model saved to models/glucotwin_model.pt (Peak Val Accuracy: {best_val_acc*100:.2f}%)")

if __name__ == "__main__":
    run_training()