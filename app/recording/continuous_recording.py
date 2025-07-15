import sounddevice as sd
import numpy as np
import wave
import os
import time
from datetime import datetime
import signal
import sys
import threading
from collections import deque

# グローバル変数でレコーダーインスタンスを管理
recorder_instance = None

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

def save_audio(recording, fs, channels, output_file):
    """音声データをWAVファイルとして保存"""
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)  # チャネル数
            wf.setsampwidth(2)         # 16bit → 2バイト
            wf.setframerate(fs)        # サンプルレート
            wf.writeframes(recording.tobytes())  # numpy 配列からバイト列に変換して書き込み
    except Exception as e:
        print(f"ファイル保存エラー: {e}")

class ContinuousRecorder:
    def __init__(self, mic_device, fs=44100, channels=1):
        self.mic_device = mic_device
        self.fs = fs
        self.channels = channels
        self.is_recording = False
        self.audio_buffer = deque(maxlen=5)  # 5秒分の1秒チャンクを保持
        self.recording_count = 0
        self.data_folder = "data"
        self.recording_thread = None
        self.save_thread = None
        self.stop_event = threading.Event()
        
        # dataフォルダを作成（存在しない場合）
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"{self.data_folder} フォルダを作成しました。")
    
    def record_chunk(self):
        """1秒間の音声チャンクを録音"""
        try:
            chunk_duration = 1  # 1秒
            recording = sd.rec(int(chunk_duration * self.fs), 
                              samplerate=self.fs, 
                              channels=self.channels, 
                              dtype='int16', 
                              device=self.mic_device)
            sd.wait()
            return recording
        except Exception as e:
            print(f"録音チャンクエラー: {e}")
            return None
    
    def continuous_chunk_recording(self):
        """1秒ごとに音声チャンクを録音し続ける"""
        while self.is_recording and not self.stop_event.is_set():
            try:
                chunk = self.record_chunk()
                if chunk is not None:
                    self.audio_buffer.append(chunk)
                
                # 短時間待機（停止要求をチェック）
                for _ in range(10):  # 0.1秒を10分割
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"録音エラー: {e}")
                break
    
    def save_last_5_seconds(self):
        """直前5秒分のデータを保存"""
        try:
            if len(self.audio_buffer) == 5:  # 5秒分のデータが揃った場合
                # 5つのチャンクを結合
                combined_audio = np.concatenate(list(self.audio_buffer))
                
                # ファイル名生成
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
                self.recording_count += 1
                output_file = os.path.join(self.data_folder, 
                                         f"recording_{timestamp}_{self.recording_count:06d}.wav")
                
                # 保存
                save_audio(combined_audio, self.fs, self.channels, output_file)
                print(f"保存: {output_file} (5秒分)")
                
                return True
        except Exception as e:
            print(f"保存エラー: {e}")
        return False
    
    def save_timer(self):
        """1秒ごとに保存処理を実行"""
        while self.is_recording and not self.stop_event.is_set():
            # 1秒待機（停止要求をチェックしながら）
            for _ in range(100):  # 1秒を100分割
                if self.stop_event.is_set():
                    break
                time.sleep(0.01)
            
            if self.is_recording and not self.stop_event.is_set():
                self.save_last_5_seconds()
    
    def start_recording(self):
        """録音開始"""
        self.is_recording = True
        self.stop_event.clear()
        
        print(f"使用デバイス: {sd.query_devices(self.mic_device)['name']}")
        print("連続録音を開始します（直前5秒分を1秒ごとに保存）")
        print("停止するには Ctrl+C を押してください\n")
        
        # 録音スレッドと保存スレッドを開始
        self.recording_thread = threading.Thread(target=self.continuous_chunk_recording)
        self.save_thread = threading.Thread(target=self.save_timer)
        
        self.recording_thread.daemon = True
        self.save_thread.daemon = True
        
        self.recording_thread.start()
        
        # 5秒待ってから保存開始（最初の5秒分のデータが溜まるまで）
        print("初期化中... 5秒後に保存開始")
        for i in range(50):  # 5秒を50分割してチェック
            if self.stop_event.is_set():
                return
            time.sleep(0.1)
        
        if not self.stop_event.is_set():
            self.save_thread.start()
        
        try:
            while self.is_recording and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_recording()
    
    def stop_recording(self):
        """録音停止"""
        print(f"\n録音停止処理を開始...")
        self.is_recording = False
        self.stop_event.set()
        
        # 進行中の録音を停止
        try:
            sd.stop()
        except:
            pass
        
        # スレッドの終了を待機
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
        
        print(f"録音を停止しました。合計 {self.recording_count} ファイルを保存しました。")

def signal_handler(sig, frame):
    """Ctrl+Cで終了時の処理"""
    global recorder_instance
    print('\n\n録音を停止しています...')
    
    if recorder_instance:
        recorder_instance.stop_recording()
    
    # 少し待ってから強制終了
    time.sleep(1)
    sys.exit(0)

def continuous_recording():
    """メイン関数"""
    global recorder_instance
    
    # USBマイクを検索
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("録音を中止しました。USBマイクが見つかりません。")
        return
    
    # Ctrl+Cでの終了処理を設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # レコーダーを作成して開始
    recorder_instance = ContinuousRecorder(mic_device)
    
    try:
        recorder_instance.start_recording()
    except Exception as e:
        print(f"録音中にエラーが発生しました: {e}")
    finally:
        if recorder_instance:
            recorder_instance.stop_recording()

if __name__ == "__main__":
    continuous_recording() 