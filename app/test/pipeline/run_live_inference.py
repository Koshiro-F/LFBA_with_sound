# run_live_inference.py

import torch
import sounddevice as sd

# 作成したモジュールをインポート
from recorder import find_microphone, record_audio
from audio_model import AudioClassifier, load_trained_model, preprocess_recording, predict

# --- パラメータ設定 ---
MODEL_PATH = "best_pytorch_model.pth"  # 学習済みモデルのパス
SAMPLE_RATE = 44100
DURATION = 5
CHANNELS = 1

def main():
    """メインの実行関数"""
    # --- 1. モデルの準備 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    model = load_trained_model(MODEL_PATH, DEVICE)
    if model is None:
        print("モデルのロードに失敗しました。プログラムを終了します。")
        return

    # --- 2. マイクの準備 ---
    mic_id = find_microphone()
    if mic_id is None:
        print("利用可能なマイクが見つかりません。プログラムを終了します。")
        return
        
    print("\n----------------------------------------")
    print(f"使用デバイス: {sd.query_devices(mic_id)['name']}")
    print("準備完了。")
    print("----------------------------------------")

    # --- 3. メインループ ---
    try:
        while True:
            user_input = input("Enterキーを押すと5秒間の録音と推論を開始します ('q'で終了): ")
            if user_input.lower() == 'q':
                break

            # 録音
            recording = record_audio(DURATION, SAMPLE_RATE, CHANNELS, mic_id)
            
            # 前処理
            input_tensor = preprocess_recording(recording, SAMPLE_RATE)
            
            # 推論
            label, confidence = predict(model, input_tensor, DEVICE)
            
            # 結果表示
            print("\n--- 💡 推論結果 ---")
            print(f"  予測ラベル: {label}")
            print(f"  信頼度: {confidence:.2%}")
            print("----------------------------------------")

    except KeyboardInterrupt:
        print("\nプログラムが中断されました。")
    finally:
        print("終了します。")


if __name__ == "__main__":
    main()