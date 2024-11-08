import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CompressModule(nn.Module):
    def __init__(self, compress_size, seq_len, d_model):
        super().__init__()
        self.compress_size = compress_size
        self.seq_len = seq_len
        
        self.model = nn.Sequential(
            nn.Linear(128, d_model),
            nn.ReLU(),
            nn.Unflatten(1, (1, compress_size * seq_len)),
            nn.Conv2d(1, 1, (compress_size, 1), stride=(compress_size, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(1,2),
        )
        
    def forward(self, x):
        # input shape: (batch_size, 128, compress_size * seq_len)
        x = x.permute(0, 2, 1) # (batch_size, compress_size * seq_len, 128)
        if x.size(1) != self.compress_size * self.seq_len:
            raise ValueError(f'Input size {x.size(1)} does not match compress_size {self.compress_size} and seq_len {self.seq_len}')
        x = self.model(x)
        
        return x

class DetectionHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        # YOLO-like anchor-free detection head  
        self.cls_center_proj = nn.Linear(d_model, d_model)
        self.cls_logits = nn.Linear(d_model, num_classes)
        self.centerness = nn.Linear(d_model, 1)
        
        self.regression_proj = nn.Linear(d_model, d_model)
        self.bbox_pred = nn.Linear(d_model, 2)
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        cls_center = self.cls_center_proj(x)
        
        classes = self.cls_logits(cls_center)
        classes = self.softmax(classes)
        
        centerness = self.centerness(cls_center)
        centerness = self.sigmoid(centerness)
        
        regression = self.regression_proj(x)
        bbox = self.bbox_pred(regression)
        bbox = torch.exp(bbox) # exp to make sure it's positive
        
        return classes, centerness, bbox



class AudioSegFormer(nn.Module):
    def __init__(self, num_classes, seq_len, d_model, num_heads, num_layers, compress_size):
        super().__init__()
        self.compress_module = CompressModule(compress_size, seq_len, d_model)
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers)
        self.detection_head = DetectionHead(d_model, num_classes)
    
    def forward(self, x):
        x = self.compress_module(x)
        x = self.encoder(x)
        classes, centerness, bbox = self.detection_head(x)
        
        return classes, centerness, bbox

if __name__ == '__main__':
    model = AudioSegFormer(num_classes=5, seq_len=10, d_model=64, num_heads=8, num_layers=4, compress_size=5)
    
    x = torch.randn(16, 128, 50)
    classes, centerness, bbox = model(x)
    print(classes.shape)
    print(centerness.shape)
    print(bbox.shape)