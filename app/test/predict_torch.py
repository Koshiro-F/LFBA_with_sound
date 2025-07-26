import argparse
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# -----------------------------------------------------------------------------
# スクリプトの説明と使い方
# -----------------------------------------------------------------------------
#
# このスクリプトは、学習済みのPyTorchモデル(.pth)を使用して、
# 指定された単一の音声ファイルから分類結果を推論します。
#
# 使い方 (ターミナルで実行):
# python predict_pytorch.py --model_path path/to/your/best_model.pth --audio_path path/to/your/audio.wav
#
# 例:
# python predict_pytorch.py --model_path best_pytorch_model.pth --audio_path my_sounds/light_on.wav
#

# -----------------------------------------------------------------------------
# パラメータ設定 (学習時と完全に一致させる必要があります)
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION_SECONDS = 5
N_MELS = 128
NUM_CLASSES = 3

# デバイス設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# モデルクラスの定義 (学習時と同一のものを記述)
# -----------------------------------------------------------------------------

class AudioClassifier(nn.Module):
    """PyTorchのカスタムモデルクラス"""
    def __init__(self, num_classes=3):
        super(AudioClassifier, self).__init__()
        # PyTorch 1.9以降では weights パラメータが推奨されています
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 1. 入力層の変更 (1チャネル -> 3チャネル)
        original_first_layer = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
             self.mobilenet.features[0][0].weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)

        # 2. 出力層の変更
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

# -----------------------------------------------------------------------------
# 前処理関数
# -----------------------------------------------------------------------------

def preprocess_audio_for_prediction(file_path):
    """
    推論用に単一の音声ファイルを前処理し、PyTorchテンソルを返す。
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION_SECONDS, res_type='kaiser_fast')
        if librosa.get_duration(y=y, sr=sr) < DURATION_SECONDS:
            y = librosa.util.fix_length(y, size=SAMPLE_RATE * DURATION_SECONDS)
        
        melspec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        # PyTorchのテンソル形式 (C, H, W) に変換
        tensor = torch.from_numpy(melspec_db).float().unsqueeze(0)
        
        # バッチ次元を追加 (B, C, H, W)
        return tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# -----------------------------------------------------------------------------
# メインの実行ブロック
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an audio file using a trained PyTorch model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the .wav audio file to classify.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit()
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        exit()

    # 1. モデルの骨格をインスタンス化
    model = AudioClassifier(num_classes=NUM_CLASSES).to(DEVICE)

    # 2. 学習済みの重み（状態辞書）をロード
    print(f"Loading model weights from: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()
        
    # 3. モデルを評価モードに設定 (重要！)
    #    Dropoutなどの層を無効化します。
    model.eval()

    # 4. 入力音声を前処理
    input_tensor = preprocess_audio_for_prediction(args.audio_path)

    if input_tensor is not None:
        input_tensor = input_tensor.to(DEVICE)

        # 5. 推論を実行 (勾配計算を無効化して高速化)
        print("Running prediction...")
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # 6. 結果を解釈して表示
        # Softmaxを適用して確率に変換
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        confidence, predicted_class_index = torch.max(probabilities, 0)
        
        class_names = {0: "OFF", 1: "ON", 2: "KEEP"}
        predicted_label = class_names.get(predicted_class_index.item(), "Unknown")
        
        print("\n--- 💡 推論結果 ---")
        print(f"  予測ラベル: {predicted_label} (クラスID: {predicted_class_index.item()})")
        print(f"  信頼度: {confidence.item():.2%}")
        print("--------------------")