import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "best_emotion_model.pt"
SEQ_LEN = 5
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
PRED_INTERVAL = 2.0  # seconds between predictions

st.set_page_config(page_title="🎥 Real-Time Emotion Recognition", layout="wide")
st.title("🎥 Real-Time Emotion Recognition")

frame_placeholder = st.empty()

# ---------------- MODEL CLASSES ----------------
# Copy your model classes from Flask server
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1536
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
        self.attn = AttentionModule(hidden_dim*2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(),
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

# ---------------- LOAD MODEL ----------------
model = CNN_AE_BiLSTM_Attn_Model(num_classes=len(CLASS_NAMES))
state = torch.load(MODEL_PATH, map_location=DEVICE)
if 'model_state_dict' in state:
    model.load_state_dict(state['model_state_dict'])
else:
    model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ---------------- FACE DETECTION ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        face = gray[y:y+h, x:x+w]
    else:
        face = gray
    face_resized = cv2.resize(face, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(face_resized)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

# ---------------- REAL-TIME WEBCAM ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot access camera")
    st.stop()

st.session_state.streaming = True
last_pred_time = time.time() - PRED_INTERVAL
current_label = "Neutral"

while st.session_state.streaming:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction only every PRED_INTERVAL seconds
    if time.time() - last_pred_time >= PRED_INTERVAL:
        face_proc = preprocess_face(frame)
        arr = np.array(face_proc).astype(np.float32)/255.0
        tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
        tensor = tensor.unsqueeze(1).repeat(1, SEQ_LEN,1,1,1).to(DEVICE)

        with torch.no_grad():
            preds = model(tensor)
            probs = F.softmax(preds, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            current_label = CLASS_NAMES[idx]

        last_pred_time = time.time()

    # Draw rectangle and label
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, current_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()
