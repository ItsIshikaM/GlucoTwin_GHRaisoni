import torch
import torch.nn as nn

class GlucoTwinFusionNet(nn.Module):
    """
    Multimodal Digital Twin Fusion Architecture:
    - Static Branch: Processes patient demographic, baseline HbA1c, and EHR records.
    - Dynamic Branch: 2-layer LSTM processing rolling 2-hour sensor windows (CGM, HRV, Steps).
    - Late Fusion Head: Concatenates latent representations to forecast 2-hour forward risk.
    """
    def __init__(self, static_dim: int = 5, dynamic_features: int = 3, lstm_hidden: int = 64):
        super().__init__()
        
        # 1. Dynamic Time-Series Processing (Wearables)
        self.lstm = nn.LSTM(
            input_size=dynamic_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 2. Static Clinical EHR Embedding
        self.ehr_dense = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # 3. Multimodal Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, static_x: torch.Tensor, dynamic_seq: torch.Tensor) -> torch.Tensor:
        # dynamic_seq shape: [Batch, Sequence_Length (24), dynamic_features (3)]
        _, (hn, _) = self.lstm(dynamic_seq)
        dynamic_latent = hn[-1]  # Extract final top-layer hidden state
        
        # static_x shape: [Batch, static_dim (5)]
        static_latent = self.ehr_dense(static_x)
        
        # Feature Concatenation & Classification
        multimodal_latent = torch.cat((dynamic_latent, static_latent), dim=1)
        risk_probability = self.fusion_head(multimodal_latent).squeeze(-1)
        return risk_probability