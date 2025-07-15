import cv2
import os
import time
from datetime import datetime
import signal
import sys
import threading

# グローバル変数でカメラインスタンスを管理
camera_instance = None

class ContinuousCamera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_capturing = False
        self.capture_count = 0
        self.data_folder = "data"
        self.capture_thread = None
        self.stop_event = threading.Event()
        
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
    
    def capture_frame(self):
        """1フレームを撮影"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("フレーム取得に失敗しました。")
                return None
        except Exception as e:
            print(f"フレーム撮影エラー: {e}")
            return None
    
    def save_image(self, frame, output_file):
        """画像を保存"""
        try:
            success = cv2.imwrite(output_file, frame)
            if success:
                print(f"保存: {output_file}")
                return True
            else:
                print(f"画像保存に失敗: {output_file}")
                return False
        except Exception as e:
            print(f"画像保存エラー: {e}")
            return False
    
    def continuous_capture(self):
        """1秒ごとに画像を撮影し続ける"""
        while self.is_capturing and not self.stop_event.is_set():
            try:
                # フレーム撮影
                frame = self.capture_frame()
                
                if frame is not None:
                    # ファイル名生成
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # ミリ秒まで
                    self.capture_count += 1
                    output_file = os.path.join(self.data_folder, 
                                             f"image_{timestamp}_{self.capture_count:06d}.jpg")
                    
                    # 保存
                    self.save_image(frame, output_file)
                
                # 1秒待機（停止要求をチェックしながら）
                for _ in range(100):  # 1秒を100分割
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"撮影エラー: {e}")
                break
    
    def start_capture(self):
        """撮影開始"""
        if not self.initialize_camera():
            return
        
        self.is_capturing = True
        self.stop_event.clear()
        
        print("連続撮影を開始します（1秒間隔）")
        print("停止するには Ctrl+C を押してください\n")
        
        # 撮影スレッドを開始
        self.capture_thread = threading.Thread(target=self.continuous_capture)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        try:
            while self.is_capturing and not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_capture()
    
    def stop_capture(self):
        """撮影停止"""
        print(f"\n撮影停止処理を開始...")
        self.is_capturing = False
        self.stop_event.set()
        
        # スレッドの終了を待機
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # カメラリソースを解放
        if self.cap is not None:
            self.cap.release()
            print("カメラリソースを解放しました。")
        
        print(f"撮影を停止しました。合計 {self.capture_count} 枚の画像を保存しました。")

def signal_handler(sig, frame):
    """Ctrl+Cで終了時の処理"""
    global camera_instance
    print('\n\n撮影を停止しています...')
    
    if camera_instance:
        camera_instance.stop_capture()
    
    # 少し待ってから強制終了
    time.sleep(1)
    sys.exit(0)

def continuous_camera():
    """メイン関数"""
    global camera_instance
    
    # Ctrl+Cでの終了処理を設定
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # カメラインスタンスを作成して開始
    camera_instance = ContinuousCamera(camera_id=0)
    
    try:
        camera_instance.start_capture()
    except Exception as e:
        print(f"撮影中にエラーが発生しました: {e}")
    finally:
        if camera_instance:
            camera_instance.stop_capture()

if __name__ == "__main__":
    continuous_camera() 