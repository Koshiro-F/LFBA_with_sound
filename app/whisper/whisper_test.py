import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
import tempfile
import os
from datetime import datetime

def find_usb_microphone():
    """USBマイクロフォンを検索して返す"""
    devices = sd.query_devices()
    print("利用可能なオーディオデバイス:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # 入力可能なデバイスのみ
            print(f"  {i}: {device['name']} (入力チャネル数: {device['max_input_channels']})")
    
    # USBマイクを自動検出
    usb_mics = []
    for i, device in enumerate(devices):
        if (device['max_input_channels'] > 0 and 
            ('usb' in device['name'].lower() or 'microphone' in device['name'].lower())):
            usb_mics.append((i, device['name']))
    
    if usb_mics:
        print(f"\n検出されたUSBマイク:")
        for device_id, name in usb_mics:
            print(f"  デバイスID {device_id}: {name}")
        return usb_mics[0][0]  # 最初に見つかったUSBマイクを返す
    else:
        print("\nUSBマイクが見つかりませんでした。利用可能な入力デバイスを確認してください。")
        return None

def save_audio_to_temp(recording, fs, channels):
    """録音した音声を一時ファイルに保存"""
    try:
        # 一時ファイルを作成
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        # WAVファイルとして保存
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16bit → 2バイト
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())
        
        return temp_filename
    except Exception as e:
        print(f"一時ファイル保存エラー: {e}")
        return None

def transcribe_audio(audio_file, model):
    """音声ファイルを文字起こし"""
    try:
        # faster-whisperのAPIを使用
        segments, info = model.transcribe(audio_file, language='ja')
        
        # セグメントからテキストを結合
        transcription = ""
        for segment in segments:
            transcription += segment.text
        
        return transcription.strip()
    except Exception as e:
        print(f"文字起こしエラー: {e}")
        return None

def main():
    print("=== Faster-Whisper 音声文字起こしテスト ===")
    
    # モデルサイズを選択
    print("Whisperモデルサイズを選択してください:")
    print("  1. tiny   (最速、精度低)")
    print("  2. base   (バランス型) [推奨]")
    print("  3. small  (少し重い、精度良)")
    print("  4. medium (重い、精度高)")
    print("  5. large-v2 (最重、最高精度)")
    
    model_sizes = {
        "1": "tiny",
        "2": "base", 
        "3": "small",
        "4": "medium",
        "5": "large-v2"
    }
    
    while True:
        try:
            choice = input("選択 (1-5, デフォルト: 2): ").strip()
            if choice == "" or choice == "2":
                model_size = "base"
                break
            elif choice in model_sizes:
                model_size = model_sizes[choice]
                break
            else:
                print("1-5の数字を入力してください。")
        except KeyboardInterrupt:
            print("\n終了します。")
            return
    
    # Faster-Whisperモデルを読み込み（初回は時間がかかります）
    print(f"Faster-Whisperモデル（{model_size}）を読み込み中...")
    try:
        # CPU使用、int8量子化で高速化
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("モデル読み込み完了!")
    except Exception as e:
        print(f"Faster-Whisperモデル読み込みエラー: {e}")
        print("以下のコマンドでFaster-Whisperをインストールしてください:")
        print("pip install faster-whisper")
        return
    
    # USBマイクを検索
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("文字起こしテストを中止しました。USBマイクが見つかりません。")
        return
    
    # 録音パラメータ
    fs = 16000       # Whisperに最適なサンプルレート
    duration = 5     # 録音時間（秒）
    channels = 1     # モノラル
    
    print(f"\n使用デバイス: {sd.query_devices(mic_device)['name']}")
    print(f"録音時間: {duration}秒")
    print(f"サンプルレート: {fs}Hz")
    print(f"モデル: {model_size}")
    print("\n何か話してください...")
    
    try:
        while True:
            # 録音開始
            print(f"\n🎤 録音開始...({duration}秒間)")
            recording = sd.rec(int(duration * fs), 
                             samplerate=fs, 
                             channels=channels, 
                             dtype='int16', 
                             device=mic_device)
            sd.wait()  # 録音終了まで待機
            print("録音終了")
            
            # 音声を一時ファイルに保存
            temp_audio_file = save_audio_to_temp(recording, fs, channels)
            
            if temp_audio_file:
                print("🤖 文字起こし中...")
                
                # Faster-Whisperで文字起こし
                transcription = transcribe_audio(temp_audio_file, model)
                
                if transcription and len(transcription.strip()) > 0:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"📝 [{timestamp}] 文字起こし結果:")
                    print(f"    「{transcription}」")
                else:
                    print("❌ 文字起こしに失敗しました（無音または認識不可）")
                
                # 一時ファイルを削除
                try:
                    os.unlink(temp_audio_file)
                except:
                    pass
            
            # 継続確認
            print("\n次の録音を開始しますか？ (Enter: 続行, q: 終了)")
            user_input = input().strip().lower()
            if user_input == 'q' or user_input == 'quit':
                break
    
    except KeyboardInterrupt:
        print("\n\nキーボード割り込みで終了します。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    
    print("文字起こしテストを終了しました。")

if __name__ == "__main__":
    main() 