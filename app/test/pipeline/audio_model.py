# audio_model.py

import torch
import torch.nn as nn
from torchvision import models
import librosa
import numpy as np

# --- パラメータ (学習時と完全に一致させる) ---
SAMPLE_RATE = 44100
DURATION_SECONDS = 5
N_MELS = 128
NUM_CLASSES = 3

# --- モデルクラスの定義 (学習時と同一) ---
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AudioClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        original_first_layer = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
             self.mobilenet.features[0][0].weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

def load_trained_model(model_path, device):
    """モデルの骨格を生成し、学習済みの重みをロードする"""
    model = AudioClassifier(num_classes=NUM_CLASSES).to(device)
    try:
        # 重みをロードし、評価モードに設定
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_recording(recording, sr):
    """
    sounddeviceで録音したNumpy配列を、モデル入力用のPyTorchテンソルに変換する
    """
    # sounddeviceからの録音データはすでにNumpy配列なので、ファイルパスからの読み込みは不要
    y = recording.astype(np.float32)

    # 長さを固定（短い場合はパディング、長い場合は切り捨て）
    target_length = SAMPLE_RATE * DURATION_SECONDS
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    elif len(y) > target_length:
        y = y[:target_length]
    
    # メルスペクトログラムに変換 & デシベル化
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # PyTorchのテンソル形式 (B, C, H, W) に変換
    tensor = torch.from_numpy(melspec_db).float().unsqueeze(0).unsqueeze(0)
    return tensor

def predict(model, input_tensor, device):
    """単一のテンソルから推論を行い、結果を返す"""
    class_names = {0: "ON", 1: "OFF", 2: "KEEP"}
    
    # 勾配計算を無効化
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        
        # Softmaxを適用して確率に変換
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 最も確率の高いクラスを取得
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_label = class_names.get(predicted_idx.item(), "Unknown")
        confidence_score = confidence.item()

    return predicted_label, confidence_score