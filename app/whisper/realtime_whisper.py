import sounddevice as sd
import numpy as np
import wave
from faster_whisper import WhisperModel
import tempfile
import os
import threading
import time
from datetime import datetime
from collections import deque
import signal
import sys

class RealtimeWhisperTranscriber:
    def __init__(self, mic_device, model_size="base", chunk_duration=3):
        self.mic_device = mic_device
        self.model_size = model_size
        self.chunk_duration = chunk_duration  # 録音チャンクの長さ（秒）
        
        # 音声設定
        self.fs = 16000  # Whisperに最適なサンプルレート
        self.channels = 1  # モノラル
        
        # 状態管理
        self.is_running = False
        self.stop_event = threading.Event()
        
        # 音声バッファ
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # スレッド
        self.recording_thread = None
        self.transcription_thread = None
        
        # Faster-Whisperモデル
        self.model = None
        
        # 結果保存
        self.transcriptions = []
        
    def load_whisper_model(self):
        """Faster-Whisperモデルを読み込み"""
        try:
            print(f"Faster-Whisperモデル（{self.model_size}）を読み込み中...")
            # CPU使用、int8量子化で高速化
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            print("モデル読み込み完了!")
            return True
        except Exception as e:
            print(f"Faster-Whisperモデル読み込みエラー: {e}")
            print("以下のコマンドでFaster-Whisperをインストールしてください:")
            print("pip install faster-whisper")
            return False
    
    def record_audio_chunk(self):
        """指定時間の音声チャンクを録音"""
        try:
            recording = sd.rec(int(self.chunk_duration * self.fs), 
                             samplerate=self.fs, 
                             channels=self.channels, 
                             dtype='int16', 
                             device=self.mic_device)
            sd.wait()
            return recording
        except Exception as e:
            print(f"録音エラー: {e}")
            return None
    
    def save_audio_to_temp(self, recording):
        """録音した音声を一時ファイルに保存"""
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(recording.tobytes())
            
            return temp_filename
        except Exception as e:
            print(f"一時ファイル保存エラー: {e}")
            return None
    
    def transcribe_audio(self, audio_file):
        """音声ファイルを文字起こし"""
        try:
            # faster-whisperのAPIを使用
            segments, info = self.model.transcribe(audio_file, language='ja')
            
            # セグメントからテキストを結合
            transcription = ""
            for segment in segments:
                transcription += segment.text
            
            return transcription.strip()
        except Exception as e:
            print(f"文字起こしエラー: {e}")
            return None
    
    def recording_loop(self):
        """録音ループ"""
        print("🎤 録音開始...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # 音声チャンクを録音
                chunk = self.record_audio_chunk()
                
                if chunk is not None:
                    # バッファに追加
                    with self.buffer_lock:
                        self.audio_buffer.append(chunk)
                
                # 短時間の待機
                if not self.stop_event.is_set():
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"録音ループエラー: {e}")
                break
    
    def transcription_loop(self):
        """文字起こしループ"""
        print("🤖 文字起こし開始...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # バッファから音声チャンクを取得
                chunk = None
                with self.buffer_lock:
                    if self.audio_buffer:
                        chunk = self.audio_buffer.popleft()
                
                if chunk is not None:
                    # 一時ファイルに保存
                    temp_file = self.save_audio_to_temp(chunk)
                    
                    if temp_file:
                        # 文字起こし実行
                        transcription = self.transcribe_audio(temp_file)
                        
                        if transcription and len(transcription.strip()) > 0:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            
                            # 結果を保存
                            result = {
                                'timestamp': timestamp,
                                'text': transcription
                            }
                            self.transcriptions.append(result)
                            
                            # 結果を表示
                            print(f"📝 [{timestamp}] {transcription}")
                        
                        # 一時ファイルを削除
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                else:
                    # バッファが空の場合は少し待機
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"文字起こしループエラー: {e}")
                break
    
    def start(self):
        """リアルタイム文字起こし開始"""
        if not self.load_whisper_model():
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        print(f"使用デバイス: {sd.query_devices(self.mic_device)['name']}")
        print(f"チャンク時間: {self.chunk_duration}秒")
        print(f"サンプルレート: {self.fs}Hz")
        print(f"モデル: {self.model_size}")
        print("\n話し始めてください... (Ctrl+C で停止)")
        print("-" * 50)
        
        # スレッド開始
        self.recording_thread = threading.Thread(target=self.recording_loop)
        self.transcription_thread = threading.Thread(target=self.transcription_loop)
        
        self.recording_thread.daemon = True
        self.transcription_thread.daemon = True
        
        self.recording_thread.start()
        self.transcription_thread.start()
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
        
        return True
    
    def stop(self):
        """リアルタイム文字起こし停止"""
        print(f"\n停止処理中...")
        
        self.is_running = False
        self.stop_event.set()
        
        # 録音を停止
        try:
            sd.stop()
        except:
            pass
        
        # スレッドの終了を待機
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=5.0)
        
        print("リアルタイム文字起こしを停止しました。")
        
        # 結果のサマリーを表示
        if self.transcriptions:
            print(f"\n=== 文字起こし結果サマリー ({len(self.transcriptions)}件) ===")
            for i, result in enumerate(self.transcriptions, 1):
                print(f"{i:2d}. [{result['timestamp']}] {result['text']}")

def find_usb_microphone():
    """USBマイクロフォンを検索して返す"""
    devices = sd.query_devices()
    print("利用可能なオーディオデバイス:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
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
        return usb_mics[0][0]
    else:
        print("\nUSBマイクが見つかりませんでした。利用可能な入力デバイスを確認してください。")
        return None

# グローバル変数でインスタンスを管理
transcriber_instance = None

def signal_handler(signum, frame):
    """シグナルハンドラー"""
    global transcriber_instance
    if transcriber_instance:
        transcriber_instance.stop()
    sys.exit(0)

def main():
    global transcriber_instance
    
    print("=== リアルタイム Faster-Whisper 文字起こしシステム ===")
    
    # USBマイクを検索
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("リアルタイム文字起こしを中止しました。USBマイクが見つかりません。")
        return
    
    # モデルサイズを選択（オプション）
    print("\nWhisperモデルサイズを選択してください:")
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
    
    # シグナルハンドラーを設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # リアルタイム文字起こしシステムを作成・開始
    transcriber_instance = RealtimeWhisperTranscriber(
        mic_device=mic_device,
        model_size=model_size,
        chunk_duration=3  # 3秒チャンク
    )
    
    try:
        transcriber_instance.start()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        if transcriber_instance:
            transcriber_instance.stop()

if __name__ == "__main__":
    main() 