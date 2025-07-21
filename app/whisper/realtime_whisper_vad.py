import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import threading
import time
from datetime import datetime
from collections import deque
import signal
import sys
import torch
import scipy.signal

class EnhancedRealtimeWhisperTranscriber:
    def __init__(self, mic_device, model_name="openai/whisper-base", chunk_duration=3, method="transformers"):
        self.mic_device = mic_device
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.method = method
        
        # 音声設定
        self.fs = 16000
        self.channels = 1
        self.nyquist_freq = self.fs / 2  # ナイキスト周波数
        
        # GPU設定
        self.device = self._check_device()
        
        # 状態管理
        self.is_running = False
        self.stop_event = threading.Event()
        
        # 音声バッファ
        self.audio_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # スレッド
        self.recording_thread = None
        self.transcription_thread = None
        
        # モデル
        self.model = None
        self.processor = None
        
        # 結果保存
        self.transcriptions = []
        
        # パフォーマンス統計
        self.processing_times = []
        
        # VADとノイズフィルタリング設定（安全な周波数範囲）
        self.vad_threshold = 0.02  # 音声検出の閾値
        self.min_speech_duration = 0.5  # 最小音声時間（秒）
        self.noise_gate_threshold = 0.01  # ノイズゲートの閾値
        self.background_noise_level = 0.005  # バックグラウンドノイズレベル
        
        # フィルタ設定（安全な周波数範囲）
        self.highpass_freq = max(50, 0.01 * self.nyquist_freq)  # 最低50Hz、またはナイキストの1%
        self.lowpass_freq = min(7000, 0.875 * self.nyquist_freq)  # 最高7kHz、またはナイキストの87.5%
        
        # ノイズプロファイル（適応的に更新）
        self.noise_profile = None
        self.noise_samples = []
        
        # 統計情報
        self.total_chunks = 0
        self.speech_chunks = 0
        self.filtered_chunks = 0
        
        print("🎯 音声アクティビティ検出（VAD）機能を有効にしました")
        print("🔇 ノイズフィルタリング機能を有効にしました")
        print(f"🔧 フィルタ周波数範囲: {self.highpass_freq:.0f}Hz - {self.lowpass_freq:.0f}Hz")
    
    def _check_device(self):
        """GPU利用可能性をチェック"""
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) が利用可能です")
            return "mps"
        elif torch.cuda.is_available():
            print("✅ NVIDIA GPU (CUDA) が利用可能です")
            return "cuda"
        else:
            print("⚠️  GPU が利用できません。CPUを使用します")
            return "cpu"
    
    def update_noise_profile(self, audio_chunk):
        """ノイズプロファイルを適応的に更新"""
        try:
            # 音声がない場合のサンプルをノイズとして学習
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            if rms < self.noise_gate_threshold:
                self.noise_samples.append(audio_chunk)
                
                # 最新50サンプルのみ保持
                if len(self.noise_samples) > 50:
                    self.noise_samples.pop(0)
                
                # ノイズプロファイルを更新
                if len(self.noise_samples) >= 5:
                    combined_noise = np.concatenate(self.noise_samples)
                    self.noise_profile = np.mean(combined_noise)
                    self.background_noise_level = np.std(combined_noise) * 2
        except Exception as e:
            print(f"ノイズプロファイル更新エラー: {e}")
    
    def apply_noise_reduction(self, audio_chunk):
        """スペクトル減算によるノイズ除去（エラーハンドリング強化）"""
        try:
            # 入力データの検証
            if len(audio_chunk) < 256:
                return audio_chunk
            
            # STFTのパラメータを安全に設定
            nperseg = min(256, len(audio_chunk) // 4)
            if nperseg < 4:
                return audio_chunk
            
            # 短時間フーリエ変換
            f, t, stft = scipy.signal.stft(audio_chunk, fs=self.fs, nperseg=nperseg)
            
            # スペクトラムの大きさと位相
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # ノイズプロファイルが存在する場合、スペクトル減算を適用
            if self.noise_profile is not None:
                # ノイズレベルの推定
                noise_power = self.background_noise_level ** 2
                signal_power = magnitude ** 2
                
                # ウィーナーフィルタの適用（安全な範囲で）
                alpha = 1.5  # 減算係数を控えめに
                enhanced_magnitude = magnitude * np.maximum(
                    0.1, 1 - alpha * noise_power / (signal_power + 1e-10)
                )
            else:
                enhanced_magnitude = magnitude
            
            # 逆短時間フーリエ変換
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            _, enhanced_audio = scipy.signal.istft(enhanced_stft, fs=self.fs, nperseg=nperseg)
            
            # 元の長さに合わせてトリミング
            if len(enhanced_audio) > len(audio_chunk):
                enhanced_audio = enhanced_audio[:len(audio_chunk)]
            elif len(enhanced_audio) < len(audio_chunk):
                # パディング
                enhanced_audio = np.pad(enhanced_audio, (0, len(audio_chunk) - len(enhanced_audio)))
            
            return enhanced_audio.astype(np.int16)
            
        except Exception as e:
            print(f"ノイズ除去エラー: {e}")
            return audio_chunk
    
    def apply_audio_filters(self, audio_chunk):
        """音声フィルタリングの適用（エラーハンドリング強化）"""
        try:
            # 入力データの検証
            if len(audio_chunk) == 0:
                return audio_chunk
            
            # 正規化
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # フィルタ周波数の妥当性をチェック
            if self.highpass_freq >= self.nyquist_freq or self.lowpass_freq >= self.nyquist_freq:
                print(f"⚠️  フィルタ周波数が範囲外です。フィルタリングをスキップします。")
                return audio_chunk
            
            # ハイパスフィルタ（低周波ノイズ除去）
            try:
                # 正規化された周波数を使用
                high_normalized = self.highpass_freq / self.nyquist_freq
                if 0 < high_normalized < 1:
                    sos_high = scipy.signal.butter(4, high_normalized, btype='high', output='sos')
                    filtered_audio = scipy.signal.sosfilt(sos_high, audio_float)
                else:
                    filtered_audio = audio_float
            except Exception as e:
                print(f"ハイパスフィルタエラー: {e}")
                filtered_audio = audio_float
            
            # ローパスフィルタ（高周波ノイズ除去）
            try:
                # 正規化された周波数を使用
                low_normalized = self.lowpass_freq / self.nyquist_freq
                if 0 < low_normalized < 1:
                    sos_low = scipy.signal.butter(4, low_normalized, btype='low', output='sos')
                    filtered_audio = scipy.signal.sosfilt(sos_low, filtered_audio)
            except Exception as e:
                print(f"ローパスフィルタエラー: {e}")
            
            # 正規化して元のスケールに戻す
            filtered_audio = np.clip(filtered_audio, -1.0, 1.0)
            return (filtered_audio * 32767).astype(np.int16)
            
        except Exception as e:
            print(f"オーディオフィルタエラー: {e}")
            return audio_chunk
    
    def detect_voice_activity(self, audio_chunk):
        """音声アクティビティ検出（VAD）"""
        try:
            # 音声を正規化
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # 1. RMS（Root Mean Square）エネルギー計算
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            # 2. ゼロクロッシング率計算
            if len(audio_float) > 1:
                zero_crossings = np.sum(np.diff(np.sign(audio_float)) != 0) / len(audio_float)
            else:
                zero_crossings = 0
            
            # 3. スペクトル重心計算
            try:
                if len(audio_float) >= 256:
                    f, psd = scipy.signal.welch(audio_float, fs=self.fs, nperseg=min(256, len(audio_float)))
                    spectral_centroid = np.sum(f * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                else:
                    spectral_centroid = 0
            except Exception as e:
                print(f"スペクトル重心計算エラー: {e}")
                spectral_centroid = 0
            
            # 4. 総合的な音声判定
            energy_check = rms > self.vad_threshold
            spectral_check = 200 < spectral_centroid < 4000  # 人間の音声周波数範囲
            zcr_check = 0.01 < zero_crossings < 0.5  # 適切なゼロクロッシング率
            
            # デバッグ情報
            voice_detected = energy_check and (spectral_check or zcr_check)
            
            return {
                'voice_detected': voice_detected,
                'rms': rms,
                'zero_crossings': zero_crossings,
                'spectral_centroid': spectral_centroid,
                'energy_check': energy_check,
                'spectral_check': spectral_check,
                'zcr_check': zcr_check
            }
            
        except Exception as e:
            print(f"VAD検出エラー: {e}")
            # エラー時はデフォルトで音声として扱う
            return {
                'voice_detected': True,
                'rms': 0,
                'zero_crossings': 0,
                'spectral_centroid': 0,
                'energy_check': False,
                'spectral_check': False,
                'zcr_check': False
            }
    
    def process_audio_chunk(self, audio_chunk):
        """音声チャンクの前処理とVAD"""
        try:
            self.total_chunks += 1
            
            # 入力データの検証
            if audio_chunk is None or len(audio_chunk) == 0:
                self.filtered_chunks += 1
                return None, {'voice_detected': False}
            
            # ノイズプロファイルの更新
            self.update_noise_profile(audio_chunk)
            
            # オーディオフィルタリング適用
            filtered_audio = self.apply_audio_filters(audio_chunk)
            
            # ノイズ除去適用
            denoised_audio = self.apply_noise_reduction(filtered_audio)
            
            # VAD実行
            vad_result = self.detect_voice_activity(denoised_audio)
            
            if vad_result['voice_detected']:
                self.speech_chunks += 1
                return denoised_audio, vad_result
            else:
                self.filtered_chunks += 1
                return None, vad_result
                
        except Exception as e:
            print(f"音声チャンク処理エラー: {e}")
            self.filtered_chunks += 1
            return None, {'voice_detected': False}
    
    def load_whisper_model(self):
        """Whisperモデルを読み込み"""
        try:
            print(f"モデル（{self.model_name}）を読み込み中...")
            print(f"使用デバイス: {self.device.upper()}")
            
            if self.method == "transformers":
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                if self.device == "mps":
                    torch.mps.empty_cache()
                
            else:  # OpenAI Whisper
                import whisper
                self.model = whisper.load_model(self.model_name.split('/')[-1], device=self.device)
                self.processor = None
            
            print("✅ モデル読み込み完了!")
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            print("以下のコマンドで必要なライブラリをインストールしてください:")
            if self.method == "transformers":
                print("pip install transformers librosa torch scipy")
            else:
                print("pip install openai-whisper torch scipy")
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
            
            # 2次元配列の場合は1次元に変換
            if recording.ndim > 1:
                recording = recording.flatten()
                
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
    
    def transcribe_audio_transformers(self, audio_file):
        """Transformersライブラリで音声ファイルを文字起こし（MPS対応）"""
        try:
            import librosa
            
            audio_data, _ = librosa.load(audio_file, sr=16000)
            
            if len(audio_data) < 0.5 * 16000:
                return ""
            
            inputs = self.processor(audio_data, return_tensors="pt", sampling_rate=16000)
            input_features = inputs.input_features.to(self.device)
            
            with torch.no_grad():
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
                predicted_ids = self.model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=448,
                    num_beams=1,
                    do_sample=False
                )
            
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription.strip()
            
        except Exception as e:
            print(f"Transformers文字起こしエラー: {e}")
            return None
    
    def transcribe_audio_whisper(self, audio_file):
        """OpenAI Whisperで音声ファイルを文字起こし"""
        try:
            result = self.model.transcribe(audio_file, language='ja')
            return result['text'].strip()
        except Exception as e:
            print(f"Whisper文字起こしエラー: {e}")
            return None
    
    def transcribe_audio(self, audio_file):
        """統一された文字起こしインターフェース"""
        if self.method == "transformers":
            return self.transcribe_audio_transformers(audio_file)
        else:
            return self.transcribe_audio_whisper(audio_file)
    
    def recording_loop(self):
        """録音ループ"""
        print("🎤 録音開始...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                chunk = self.record_audio_chunk()
                
                if chunk is not None:
                    # 音声前処理とVAD
                    processed_chunk, vad_result = self.process_audio_chunk(chunk)
                    
                    if processed_chunk is not None:
                        # 音声が検出された場合のみバッファに追加
                        with self.buffer_lock:
                            self.audio_buffer.append(processed_chunk)
                            if len(self.audio_buffer) > 10:
                                self.audio_buffer.popleft()
                    else:
                        # 音声が検出されなかった場合の詳細ログ（オプション）
                        if self.total_chunks % 20 == 0:  # 20チャンクごとに統計表示
                            speech_rate = (self.speech_chunks / self.total_chunks) * 100
                            print(f"📊 VAD統計: 音声率 {speech_rate:.1f}% ({self.speech_chunks}/{self.total_chunks})")
                
                if not self.stop_event.is_set():
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"録音ループエラー: {e}")
                time.sleep(1)  # エラー時は少し待機
    
    def transcription_loop(self):
        """文字起こしループ"""
        print("🤖 文字起こし開始...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                chunk = None
                with self.buffer_lock:
                    if self.audio_buffer:
                        chunk = self.audio_buffer.popleft()
                
                if chunk is not None:
                    start_time = time.time()
                    
                    temp_file = self.save_audio_to_temp(chunk)
                    
                    if temp_file:
                        transcription = self.transcribe_audio(temp_file)
                        
                        process_time = time.time() - start_time
                        self.processing_times.append(process_time)
                        
                        if len(self.processing_times) > 50:
                            self.processing_times.pop(0)
                        
                        # 文字起こし結果のフィルタリング
                        if transcription and len(transcription.strip()) > 0:
                            # 無意味な結果をフィルタリング
                            if self.is_meaningful_transcription(transcription):
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                
                                result = {
                                    'timestamp': timestamp,
                                    'text': transcription,
                                    'process_time': process_time
                                }
                                self.transcriptions.append(result)
                                
                                avg_time = sum(self.processing_times[-10:]) / min(len(self.processing_times), 10)
                                print(f"📝 [{timestamp}] ({process_time:.2f}s, 平均: {avg_time:.2f}s) {transcription}")
                            else:
                                print(f"🔇 [{datetime.now().strftime('%H:%M:%S')}] フィルタリング済み: \"{transcription}\"")
                        
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                        
                        if self.device == "mps" and len(self.transcriptions) % 10 == 0:
                            torch.mps.empty_cache()
                            
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"文字起こしループエラー: {e}")
                if self.device == "mps":
                    torch.mps.empty_cache()
                time.sleep(1)  # エラー時は少し待機
    
    def is_meaningful_transcription(self, text):
        """意味のある文字起こし結果かを判定"""
        if not text or len(text.strip()) < 2:
            return False
        
        # 繰り返しパターンのチェック
        if len(set(text.replace(' ', ''))) < 3:  # 文字の種類が少なすぎる
            return False
        
        # 一般的なノイズ文字列のフィルタリング
        noise_patterns = [
            'ありがとうございました',  # よくある誤認識
            'お疲れ様でした',
            'はい',
            'あー',
            'えー',
            'うー',
            'んー',
            '。',
            '、',
            'あ',
            'え',
            'う',
            'お'
        ]
        
        text_lower = text.lower().strip()
        if text_lower in noise_patterns or len(text_lower) <= 2:
            return False
        
        return True
    
    def start(self):
        """リアルタイム文字起こし開始"""
        if not self.load_whisper_model():
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        print(f"🎙️  使用デバイス: {sd.query_devices(self.mic_device)['name']}")
        print(f"⏱️  チャンク時間: {self.chunk_duration}秒")
        print(f"🔊 サンプルレート: {self.fs}Hz")
        print(f"🧠 モデル: {self.model_name}")
        print(f"🖥️  実行環境: {self.device.upper()}")
        print(f"⚙️  実装方式: {self.method}")
        print(f"🎯 VAD閾値: {self.vad_threshold}")
        print(f"🔇 ノイズゲート: {self.noise_gate_threshold}")
        print("\n話し始めてください... (Ctrl+C で停止)")
        print("💡 無音時やノイズのみの場合は文字起こしをスキップします")
        print("-" * 70)
        
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
        
        try:
            sd.stop()
        except:
            pass
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=5.0)
        
        if self.device == "mps":
            torch.mps.empty_cache()
        
        print("✅ リアルタイム文字起こしを停止しました。")
        
        # 詳細統計を表示
        if self.total_chunks > 0:
            speech_rate = (self.speech_chunks / self.total_chunks) * 100
            filter_rate = (self.filtered_chunks / self.total_chunks) * 100
            
            print(f"\n📊 VAD統計:")
            print(f"  総チャンク数: {self.total_chunks}")
            print(f"  音声検出: {self.speech_chunks} ({speech_rate:.1f}%)")
            print(f"  フィルタリング: {self.filtered_chunks} ({filter_rate:.1f}%)")
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
            
            print(f"\n📊 パフォーマンス統計:")
            print(f"  平均処理時間: {avg_time:.2f}秒")
            print(f"  最短処理時間: {min_time:.2f}秒")
            print(f"  最長処理時間: {max_time:.2f}秒")
            print(f"  総処理回数: {len(self.processing_times)}回")
        
        if self.transcriptions:
            print(f"\n=== 文字起こし結果サマリー ({len(self.transcriptions)}件) ===")
            for i, result in enumerate(self.transcriptions[-10:], 1):
                print(f"{i:2d}. [{result['timestamp']}] ({result['process_time']:.2f}s) {result['text']}")
            
            if len(self.transcriptions) > 10:
                print(f"... 他 {len(self.transcriptions) - 10} 件")

def find_usb_microphone():
    """USBマイクロフォンを検索して返す"""
    devices = sd.query_devices()
    print("利用可能なオーディオデバイス:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (入力チャネル数: {device['max_input_channels']})")
    
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

transcriber_instance = None

def signal_handler(signum, frame):
    global transcriber_instance
    if transcriber_instance:
        transcriber_instance.stop()
    sys.exit(0)

def main():
    global transcriber_instance
    
    print("=== VAD & ノイズフィルタリング対応 リアルタイム Whisper 文字起こしシステム (修正版) ===")
    
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("リアルタイム文字起こしを中止しました。USBマイクが見つかりません。")
        return
    
    # 実装方式を選択
    print("\n実装方式を選択してください:")
    print("  1. OpenAI Whisper (CPU/MPS) - 安定")
    print("  2. Transformers + MPS (GPU) - 高速 [推奨]")
    
    while True:
        try:
            method_choice = input("選択 (1-2, デフォルト: 2): ").strip()
            if method_choice == "" or method_choice == "2":
                method = "transformers"
                break
            elif method_choice == "1":
                method = "whisper"
                break
            else:
                print("1-2の数字を入力してください。")
        except KeyboardInterrupt:
            print("\n終了します。")
            return
    
    # モデルサイズを選択
    print("\nWhisperモデルサイズを選択してください:")
    if method == "transformers":
        print("  1. tiny   (最速、精度低)")
        print("  2. base   (バランス型) [推奨]")
        print("  3. small  (少し重い、精度良)")
        print("  4. medium (重い、精度高)")
        print("  5. large-v2 (最重、最高精度)")
        print("  6. large-v3 (最新、最高精度) [MPS推奨]")
        
        model_names = {
            "1": "openai/whisper-tiny",
            "2": "openai/whisper-base",
            "3": "openai/whisper-small",
            "4": "openai/whisper-medium",
            "5": "openai/whisper-large-v2",
            "6": "openai/whisper-large-v3"
        }
    else:
        print("  1. tiny   (最速、精度低)")
        print("  2. base   (バランス型) [推奨]")
        print("  3. small  (少し重い、精度良)")
        print("  4. medium (重い、精度高)")
        print("  5. large  (最重、最高精度)")
        
        model_names = {
            "1": "tiny",
            "2": "base",
            "3": "small",
            "4": "medium",
            "5": "large"
        }
    
    while True:
        try:
            choice = input("選択 (デフォルト: 2): ").strip()
            if choice == "" or choice == "2":
                model_name = model_names["2"]
                break
            elif choice in model_names:
                model_name = model_names[choice]
                break
            else:
                max_choice = len(model_names)
                print(f"1-{max_choice}の数字を入力してください。")
        except KeyboardInterrupt:
            print("\n終了します。")
            return
    
    # チャンク時間を選択
    print("\n音声チャンク時間を選択してください:")
    print("  1. 2秒 (高反応、低精度)")
    print("  2. 3秒 (バランス型) [推奨]")
    print("  3. 5秒 (低反応、高精度)")
    
    chunk_durations = {"1": 2, "2": 3, "3": 5}
    
    while True:
        try:
            chunk_choice = input("選択 (1-3, デフォルト: 2): ").strip()
            if chunk_choice == "" or chunk_choice == "2":
                chunk_duration = 3
                break
            elif chunk_choice in chunk_durations:
                chunk_duration = chunk_durations[chunk_choice]
                break
            else:
                print("1-3の数字を入力してください。")
        except KeyboardInterrupt:
            print("\n終了します。")
            return
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    transcriber_instance = EnhancedRealtimeWhisperTranscriber(
        mic_device=mic_device,
        model_name=model_name,
        chunk_duration=chunk_duration,
        method=method
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