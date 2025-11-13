import io
import base64
from PIL import Image
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "best_emotion_model.pt"
SEQ_LEN = 5
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# ---------------- MODEL CLASSES ----------------
class CNNEncoder(nn.Module):
    def __init__(self, train_backbone=False):
        super().__init__()
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1536
        if not train_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

class DeepFeatureAutoencoder(nn.Module):
    def __init__(self, input_dim=1536, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_out):
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context, attn_weights

class BiLSTM_Attn_Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.attn = AttentionModule(hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
    def forward(self, z_seq):
        lstm_out, _ = self.lstm(z_seq)
        context, _ = self.attn(lstm_out)
        out = self.fc(context)
        return out

class CNN_AE_BiLSTM_Attn_Model(nn.Module):
    def __init__(self, num_classes=6, latent_dim=128):
        super().__init__()
        self.cnn = CNNEncoder()
        self.ae = DeepFeatureAutoencoder(input_dim=self.cnn.out_dim, latent_dim=latent_dim)
        self.classifier = BiLSTM_Attn_Classifier(latent_dim, num_classes)
    def forward(self, frames):
        B, T, C, H, W = frames.size()
        z_seq = []
        for t in range(T):
            feats = self.cnn(frames[:, t])
            _, z = self.ae(feats)
            z_seq.append(z)
        z_seq = torch.stack(z_seq, dim=1)
        preds = self.classifier(z_seq)
        return preds

# ---------------- CLASSES ----------------
CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------- LOAD MODEL ----------------
print("Loading model on", DEVICE)
model = CNN_AE_BiLSTM_Attn_Model(num_classes=NUM_CLASSES).to(DEVICE)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
        print("Loaded model weights from", MODEL_PATH)
    except Exception as e:
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            print("Loaded model_state_dict from checkpoint")
        else:
            print("Warning: failed to load state_dict:", e)
else:
    print("Warning: model weights not found, running with random weights.")

model.eval()

# ---------------- UTILS ----------------
def preprocess_pil(img_pil, img_size=IMG_SIZE):
    img = img_pil.convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor

def pil_from_base64(b64str):
    b = base64.b64decode(b64str)
    return Image.open(io.BytesIO(b))

# ---------------- ROUTE ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'image' not in data:
        return jsonify({'error': 'No image field in JSON'}), 400

    b64 = data['image']
    if b64.startswith('data:'):
        b64 = b64.split(',',1)[1]

    try:
        pil = pil_from_base64(b64)
    except Exception as e:
        return jsonify({'error': f'Failed to decode image: {e}'}), 400

    inp = preprocess_pil(pil)
    seq = inp.unsqueeze(1).repeat(1, SEQ_LEN, 1, 1, 1).to(DEVICE)

    with torch.no_grad():
        preds = model(seq)
        probs = torch.softmax(preds, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx]

    return jsonify({
        'label': label,
        'index': idx,
        'probabilities': probs.tolist(),
        'class_names': CLASS_NAMES
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
