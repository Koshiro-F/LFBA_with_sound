import sounddevice as sd
import cv2
import numpy as np
import wave
import os
import time
from datetime import datetime
import signal
import sys
import threading
from collections import deque

# グローバル変数でインスタンスを管理
dataset_collector = None

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
        return True
    except Exception as e:
        print(f"音声保存エラー: {e}")
        return False

def save_image(frame, output_file):
    """画像を保存"""
    try:
        success = cv2.imwrite(output_file, frame)
        return success
    except Exception as e:
        print(f"画像保存エラー: {e}")
        return False

class DatasetCollector:
    def __init__(self, mic_device, camera_id=0, fs=44100, channels=1):
        self.mic_device = mic_device
        self.camera_id = camera_id
        self.fs = fs
        self.channels = channels
        
        # 状態管理
        self.is_collecting = False
        self.stop_event = threading.Event()
        
        # データ管理
        self.audio_buffer = deque(maxlen=5)  # 5秒分の1秒チャンクを保持
        self.collection_count = 0
        self.data_folder = "data"
        
        # カメラ
        self.cap = None
        
        # スレッド
        self.audio_thread = None
        self.camera_thread = None
        self.save_thread = None
        
        # dataフォルダを作成（存在しない場合）
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"{self.data_folder} フォルダを作成しました。")
    
    def initialize_camera(self):
        """カメラを初期化"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"カメラ {self.camera_id} を開けませんでした。")
                return False
            
            # カメラの設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"カメラ {self.camera_id} を初期化しました。")
            return True
        except Exception as e:
            print(f"カメラ初期化エラー: {e}")
            return False
    
    def record_audio_chunk(self):
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
            print(f"音声録音エラー: {e}")
            return None
    
    def capture_frame(self):
        """1フレームを撮影"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                return None
        except Exception as e:
            print(f"フレーム撮影エラー: {e}")
            return None
    
    def continuous_audio_recording(self):
        """1秒ごとに音声チャンクを録音し続ける"""
        while self.is_collecting and not self.stop_event.is_set():
            try:
                chunk = self.record_audio_chunk()
                if chunk is not None:
                    self.audio_buffer.append(chunk)
                
                # 短時間待機（停止要求をチェック）
                for _ in range(10):  # 0.1秒を10分割
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"音声録音エラー: {e}")
                break
    
    def create_timestamp_folder(self):
        """タイムスタンプフォルダを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.join(self.data_folder, timestamp)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        return folder_path, timestamp
    
    def save_current_data(self):
        """現在のデータを保存（音声5秒分+画像1枚）"""
        try:
            # タイムスタンプフォルダを作成
            folder_path, timestamp = self.create_timestamp_folder()
            self.collection_count += 1
            
            saved_items = []
            
            # 音声データを保存（直前5秒分）
            if len(self.audio_buffer) == 5:
                combined_audio = np.concatenate(list(self.audio_buffer))
                audio_file = os.path.join(folder_path, f"audio_{timestamp}.wav")
                if save_audio(combined_audio, self.fs, self.channels, audio_file):
                    saved_items.append("音声")
            
            # 画像を撮影・保存
            frame = self.capture_frame()
            if frame is not None:
                image_file = os.path.join(folder_path, f"image_{timestamp}.jpg")
                if save_image(frame, image_file):
                    saved_items.append("画像")
            
            if saved_items:
                print(f"保存 ({self.collection_count:04d}): {folder_path} ({', '.join(saved_items)})")
            
            return True
            
        except Exception as e:
            print(f"データ保存エラー: {e}")
            return False
    
    def data_save_timer(self):
        """1秒ごとにデータ保存処理を実行"""
        while self.is_collecting and not self.stop_event.is_set():
            # 1秒待機（停止要求をチェックしながら）
            for _ in range(100):  # 1秒を100分割
                if self.stop_event.is_set():
                    break
                time.sleep(0.01)
            
            if self.is_collecting and not self.stop_event.is_set():
                self.save_current_data()
    
    def start_collection(self):
        """データ収集開始"""
        # カメラを初期化
        if not self.initialize_camera():
            return
        
        self.is_collecting = True
        self.stop_event.clear()
        
        print(f"使用マイク: {sd.query_devices(self.mic_device)['name']}")
        print(f"使用カメラ: ID {self.camera_id}")
        print("データセット収集を開始します（音声5秒+画像1枚を1秒ごと）")
        print("停止するには Ctrl+C を押してください\n")
        
        # 各スレッドを開始
        self.audio_thread = threading.Thread(target=self.continuous_audio_recording)
        self.save_thread = threading.Thread(target=self.data_save_timer)
        
        self.audio_thread.daemon = True
        self.save_thread.daemon = True
        
        # 音声録音開始
        self.audio_thread.start()
        
        # 5秒待ってから保存開始（最初の5秒分のデータが溜まるまで）
        print("初期化中... 5秒後に保存開始")
        for i in range(50):  # 5秒を50分割してチェック
            if self.stop_event.is_set():
                return
            time.sleep(0.1)
        
        if not self.stop_event.is_set():
            self.save_thread.start()
            print("データ収集開始！\n")
        
        try:
            while self.is_collecting and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_collection()
    
    def stop_collection(self):
        """データ収集停止"""
        print(f"\nデータ収集停止処理を開始...")
        self.is_collecting = False
        self.stop_event.set()
        
        # 進行中の録音を停止
        try:
            sd.stop()
        except:
            pass
        
        # スレッドの終了を待機
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
        
        # カメラリソースを解放
        if self.cap is not None:
            self.cap.release()
            print("カメラリソースを解放しました。")
        
        print(f"データ収集を停止しました。合計 {self.collection_count} セットのデータを保存しました。")

def signal_handler(sig, frame):
    """Ctrl+Cで終了時の処理"""
    global dataset_collector
    print('\n\nデータ収集を停止しています...')
    
    if dataset_collector:
        dataset_collector.stop_collection()
    
    # 少し待ってから強制終了
    time.sleep(1)
    sys.exit(0)

def main():
    """メイン関数"""
    global dataset_collector
    
    print("=== マルチモーダルデータセット収集システム ===")
    
    # USBマイクを検索
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("データ収集を中止しました。USBマイクが見つかりません。")
        return
    
    # Ctrl+Cでの終了処理を設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # データセットコレクターを作成して開始
    dataset_collector = DatasetCollector(mic_device, camera_id=0)
    
    try:
        dataset_collector.start_collection()
    except Exception as e:
        print(f"データ収集中にエラーが発生しました: {e}")
    finally:
        if dataset_collector:
            dataset_collector.stop_collection()

if __name__ == "__main__":
    main() 