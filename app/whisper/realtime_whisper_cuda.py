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
import platform

class EnhancedRealtimeWhisperTranscriber:
    def __init__(self, mic_device, model_name="openai/whisper-base", chunk_duration=3, method="transformers"):
        self.mic_device = mic_device
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.method = method
        
        # OS検出
        self.os_name = platform.system()
        print(f"🖥️  検出されたOS: {self.os_name}")
        
        # 音声設定（動的に調整）
        self.target_fs = 16000  # Whisperに最適なサンプルレート
        self.fs = self._detect_supported_sample_rate()  # 実際の録音サンプルレート
        self.channels = 1
        self.nyquist_freq = self.fs / 2
        
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
        self.vad_threshold = 0.02
        self.min_speech_duration = 0.5
        self.noise_gate_threshold = 0.01
        self.background_noise_level = 0.005
        
        # フィルタ設定（動的に調整）
        self.highpass_freq = max(50, 0.01 * self.nyquist_freq)
        self.lowpass_freq = min(min(7000, self.target_fs//2 - 100), 0.875 * self.nyquist_freq)
        
        # ノイズプロファイル
        self.noise_profile = None
        self.noise_samples = []
        
        # 統計情報
        self.total_chunks = 0
        self.speech_chunks = 0
        self.filtered_chunks = 0
        
        print("🎯 音声アクティビティ検出（VAD）機能を有効にしました")
        print("🔇 ノイズフィルタリング機能を有効にしました")
        print(f"🔧 フィルタ周波数範囲: {self.highpass_freq:.0f}Hz - {self.lowpass_freq:.0f}Hz")
        print(f"🎙️  録音サンプルレート: {self.fs}Hz → Whisper用: {self.target_fs}Hz")
    
    def _detect_supported_sample_rate(self):
        """サポートされているサンプルレートを検出"""
        # 一般的なサンプルレートを試行順にリスト
        sample_rates = [16000, 44100, 48000, 22050, 32000, 8000]
        
        print("🔍 サポートされているサンプルレートを検出中...")
        
        for rate in sample_rates:
            try:
                # テスト録音（非常に短時間）
                test_duration = 0.1  # 100ms
                test_recording = sd.rec(
                    int(test_duration * rate), 
                    samplerate=rate, 
                    channels=1, 
                    dtype='int16', 
                    device=self.mic_device
                )
                sd.wait()
                
                print(f"✅ サンプルレート {rate}Hz: サポート済み")
                return rate
                
            except Exception as e:
                print(f"❌ サンプルレート {rate}Hz: 非サポート ({str(e)[:50]}...)")
                continue
        
        # どのサンプルレートも動作しない場合
        print("⚠️  サポートされているサンプルレートが見つかりませんでした。デフォルト設定を使用します。")
        return 16000
    
    def _check_device(self):
        """GPU利用可能性をチェック（CUDA優先）"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"✅ NVIDIA GPU (CUDA) が利用可能です")
            print(f"   GPU数: {gpu_count}, GPU名: {gpu_name}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) が利用可能です")
            return "mps"
        else:
            print("⚠️  GPU が利用できません。CPUを使用します")
            return "cpu"
    
    def resample_audio(self, audio_data, original_fs, target_fs):
        """音声のリサンプリング"""
        if original_fs == target_fs:
            return audio_data
        
        try:
            # scipy.signalを使用してリサンプリング
            num_samples = int(len(audio_data) * target_fs / original_fs)
            resampled = scipy.signal.resample(audio_data, num_samples)
            return resampled.astype(np.int16)
        except Exception as e:
            print(f"リサンプリングエラー: {e}")
            return audio_data
    
    def update_noise_profile(self, audio_chunk):
        """ノイズプロファイルを適応的に更新"""
        try:
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
            
            if rms < self.noise_gate_threshold:
                self.noise_samples.append(audio_chunk)
                
                if len(self.noise_samples) > 50:
                    self.noise_samples.pop(0)
                
                if len(self.noise_samples) >= 5:
                    combined_noise = np.concatenate(self.noise_samples)
                    self.noise_profile = np.mean(combined_noise)
                    self.background_noise_level = np.std(combined_noise) * 2
        except Exception as e:
            print(f"ノイズプロファイル更新エラー: {e}")
    
    def apply_noise_reduction(self, audio_chunk):
        """スペクトル減算によるノイズ除去"""
        try:
            if len(audio_chunk) < 256:
                return audio_chunk
            
            nperseg = min(256, len(audio_chunk) // 4)
            if nperseg < 4:
                return audio_chunk
            
            f, t, stft = scipy.signal.stft(audio_chunk, fs=self.target_fs, nperseg=nperseg)
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            if self.noise_profile is not None:
                noise_power = self.background_noise_level ** 2
                signal_power = magnitude ** 2
                
                alpha = 1.5
                enhanced_magnitude = magnitude * np.maximum(
                    0.1, 1 - alpha * noise_power / (signal_power + 1e-10)
                )
            else:
                enhanced_magnitude = magnitude
            
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            _, enhanced_audio = scipy.signal.istft(enhanced_stft, fs=self.target_fs, nperseg=nperseg)
            
            if len(enhanced_audio) > len(audio_chunk):
                enhanced_audio = enhanced_audio[:len(audio_chunk)]
            elif len(enhanced_audio) < len(audio_chunk):
                enhanced_audio = np.pad(enhanced_audio, (0, len(audio_chunk) - len(enhanced_audio)))
            
            return enhanced_audio.astype(np.int16)
            
        except Exception as e:
            print(f"ノイズ除去エラー: {e}")
            return audio_chunk
    
    def apply_audio_filters(self, audio_chunk):
        """音声フィルタリングの適用"""
        try:
            if len(audio_chunk) == 0:
                return audio_chunk
            
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # 現在のサンプルレートに基づいてナイキスト周波数を計算
            current_nyquist = self.target_fs / 2
            
            if self.highpass_freq >= current_nyquist or self.lowpass_freq >= current_nyquist:
                print(f"⚠️  フィルタ周波数が範囲外です。フィルタリングをスキップします。")
                return audio_chunk
            
            # ハイパスフィルタ
            try:
                high_normalized = self.highpass_freq / current_nyquist
                if 0 < high_normalized < 1:
                    sos_high = scipy.signal.butter(4, high_normalized, btype='high', output='sos')
                    filtered_audio = scipy.signal.sosfilt(sos_high, audio_float)
                else:
                    filtered_audio = audio_float
            except Exception as e:
                print(f"ハイパスフィルタエラー: {e}")
                filtered_audio = audio_float
            
            # ローパスフィルタ
            try:
                low_normalized = self.lowpass_freq / current_nyquist
                if 0 < low_normalized < 1:
                    sos_low = scipy.signal.butter(4, low_normalized, btype='low', output='sos')
                    filtered_audio = scipy.signal.sosfilt(sos_low, filtered_audio)
            except Exception as e:
                print(f"ローパスフィルタエラー: {e}")
            
            filtered_audio = np.clip(filtered_audio, -1.0, 1.0)
            return (filtered_audio * 32767).astype(np.int16)
            
        except Exception as e:
            print(f"オーディオフィルタエラー: {e}")
            return audio_chunk
    
    def detect_voice_activity(self, audio_chunk):
        """音声アクティビティ検出（VAD）"""
        try:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            if len(audio_float) > 1:
                zero_crossings = np.sum(np.diff(np.sign(audio_float)) != 0) / len(audio_float)
            else:
                zero_crossings = 0
            
            try:
                if len(audio_float) >= 256:
                    f, psd = scipy.signal.welch(audio_float, fs=self.target_fs, nperseg=min(256, len(audio_float)))
                    spectral_centroid = np.sum(f * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
                else:
                    spectral_centroid = 0
            except Exception as e:
                print(f"スペクトル重心計算エラー: {e}")
                spectral_centroid = 0
            
            energy_check = rms > self.vad_threshold
            spectral_check = 200 < spectral_centroid < 4000
            zcr_check = 0.01 < zero_crossings < 0.5
            
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
            
            if audio_chunk is None or len(audio_chunk) == 0:
                self.filtered_chunks += 1
                return None, {'voice_detected': False}
            
            # リサンプリング（必要な場合）
            if self.fs != self.target_fs:
                audio_chunk = self.resample_audio(audio_chunk, self.fs, self.target_fs)
            
            self.update_noise_profile(audio_chunk)
            filtered_audio = self.apply_audio_filters(audio_chunk)
            denoised_audio = self.apply_noise_reduction(filtered_audio)
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
                
                # GPU設定
                if self.device == "cuda":
                    self.model = self.model.cuda()
                    print(f"🚀 モデルをCUDA GPU({torch.cuda.get_device_name(0)})に移動しました")
                elif self.device == "mps":
                    self.model = self.model.to("mps")
                    print("🚀 モデルをApple Silicon GPUに移動しました")
                else:
                    print("💻 CPUでモデルを実行します")
                
                self.model.eval()
                
                # メモリ最適化
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
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
                if self.device == "cuda":
                    print("CUDA版PyTorchについては以下を参照してください:")
                    print("https://pytorch.org/get-started/locally/")
            else:
                print("pip install openai-whisper torch scipy")
            return False
    
    def record_audio_chunk(self):
        """指定時間の音声チャンクを録音（Ubuntu対応）"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Ubuntu環境での録音最適化
                if self.os_name == "Linux":
                    # Linuxでの録音設定を調整
                    recording = sd.rec(
                        int(self.chunk_duration * self.fs), 
                        samplerate=self.fs, 
                        channels=self.channels, 
                        dtype='int16', 
                        device=self.mic_device,
                        blocking=True  # Linuxでの安定性向上
                    )
                else:
                    recording = sd.rec(
                        int(self.chunk_duration * self.fs), 
                        samplerate=self.fs, 
                        channels=self.channels, 
                        dtype='int16', 
                        device=self.mic_device
                    )
                    sd.wait()
                
                # 2次元配列の場合は1次元に変換
                if recording.ndim > 1:
                    recording = recording.flatten()
                    
                return recording
                
            except Exception as e:
                print(f"録音エラー (試行 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # 別のサンプルレートを試行
                    if self.fs == 16000:
                        self.fs = 44100
                    elif self.fs == 44100:
                        self.fs = 48000
                    else:
                        self.fs = 16000
                    
                    self.nyquist_freq = self.fs / 2
                    print(f"サンプルレートを {self.fs}Hz に変更して再試行...")
                    time.sleep(1)
                else:
                    print("❌ 録音に失敗しました。オーディオデバイスの設定を確認してください。")
                    if self.os_name == "Linux":
                        print("Ubuntu/Linuxの場合、以下のコマンドを試してください:")
                        print("sudo apt-get install portaudio19-dev python3-pyaudio")
                        print("pulseaudio --start")
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
                wf.setframerate(self.target_fs)  # Whisper用サンプルレートで保存
                wf.writeframes(recording.tobytes())
            
            return temp_filename
        except Exception as e:
            print(f"一時ファイル保存エラー: {e}")
            return None
    
    def transcribe_audio_transformers(self, audio_file):
        """Transformersライブラリで音声ファイルを文字起こし（CUDA/MPS対応）"""
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
                    processed_chunk, vad_result = self.process_audio_chunk(chunk)
                    
                    if processed_chunk is not None:
                        with self.buffer_lock:
                            self.audio_buffer.append(processed_chunk)
                            if len(self.audio_buffer) > 10:
                                self.audio_buffer.popleft()
                    else:
                        if self.total_chunks % 20 == 0:
                            speech_rate = (self.speech_chunks / self.total_chunks) * 100
                            print(f"📊 VAD統計: 音声率 {speech_rate:.1f}% ({self.speech_chunks}/{self.total_chunks})")
                
                if not self.stop_event.is_set():
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"録音ループエラー: {e}")
                time.sleep(1)
    
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
                        
                        if transcription and len(transcription.strip()) > 0:
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
                        
                        # GPU メモリ管理
                        if self.device == "cuda" and len(self.transcriptions) % 10 == 0:
                            torch.cuda.empty_cache()
                        elif self.device == "mps" and len(self.transcriptions) % 10 == 0:
                            torch.mps.empty_cache()
                            
                else:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"文字起こしループエラー: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()
                time.sleep(1)
    
    def is_meaningful_transcription(self, text):
        """意味のある文字起こし結果かを判定"""
        if not text or len(text.strip()) < 2:
            return False
        
        if len(set(text.replace(' ', ''))) < 3:
            return False
        
        noise_patterns = [
            'ありがとうございました', 'お疲れ様でした', 'はい', 'あー', 'えー', 'うー', 'んー',
            '。', '、', 'あ', 'え', 'う', 'お', 'い'
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
        
        device_info = sd.query_devices(self.mic_device)
        print(f"🎙️  使用デバイス: {device_info['name']}")
        print(f"⏱️  チャンク時間: {self.chunk_duration}秒")
        print(f"🔊 録音サンプルレート: {self.fs}Hz")
        print(f"🎯 Whisperサンプルレート: {self.target_fs}Hz")
        print(f"🧠 モデル: {self.model_name}")
        print(f"🖥️  実行環境: {self.device.upper()}")
        if self.device == "cuda":
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        print(f"⚙️  実装方式: {self.method}")
        print(f"🎯 VAD閾値: {self.vad_threshold}")
        print(f"🔇 ノイズゲート: {self.noise_gate_threshold}")
        print("\n話し始めてください... (Ctrl+C で停止)")
        print("💡 無音時やノイズのみの場合は文字起こしをスキップします")
        print("-" * 80)
        
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
        
        # GPU メモリクリア
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
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
    """USBマイクロフォンを検索して返す（Ubuntu対応）"""
    try:
        devices = sd.query_devices()
        print("利用可能なオーディオデバイス:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (入力チャネル数: {device['max_input_channels']}, "
                      f"サンプルレート: {device['default_samplerate']}Hz)")
        
        # USBマイクを自動検出
        usb_mics = []
        for i, device in enumerate(devices):
            if (device['max_input_channels'] > 0 and 
                ('usb' in device['name'].lower() or 'microphone' in device['name'].lower() or
                 'webcam' in device['name'].lower() or 'headset' in device['name'].lower())):
                usb_mics.append((i, device['name']))
        
        if usb_mics:
            print(f"\n検出されたUSBマイク:")
            for device_id, name in usb_mics:
                print(f"  デバイスID {device_id}: {name}")
            return usb_mics[0][0]
        else:
            print("\nUSBマイクが見つかりませんでした。利用可能な入力デバイスを確認してください。")
            
            # Ubuntu/Linuxの場合の追加情報
            if platform.system() == "Linux":
                print("\nUbuntu/Linuxでの確認事項:")
                print("1. マイクが正しく接続されているか確認")
                print("2. 以下のコマンドでオーディオシステムを確認:")
                print("   arecord -l")
                print("   pulseaudio --check")
                print("3. 必要に応じて以下をインストール:")
                print("   sudo apt-get install portaudio19-dev python3-pyaudio pulseaudio")
            
            return None
            
    except Exception as e:
        print(f"オーディオデバイス検出エラー: {e}")
        return None

transcriber_instance = None

def signal_handler(signum, frame):
    global transcriber_instance
    if transcriber_instance:
        transcriber_instance.stop()
    sys.exit(0)

def main():
    global transcriber_instance
    
    print("=== CUDA対応 VAD & ノイズフィルタリング リアルタイム Whisper 文字起こしシステム ===")
    print(f"実行環境: {platform.system()} {platform.release()}")
    
    mic_device = find_usb_microphone()
    
    if mic_device is None:
        print("リアルタイム文字起こしを中止しました。USBマイクが見つかりません。")
        return
    
    # 実装方式を選択
    print("\n実装方式を選択してください:")
    print("  1. OpenAI Whisper (CPU/GPU) - 安定")
    print("  2. Transformers + GPU (CUDA/MPS) - 高速 [推奨]")
    
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
        print("  6. large-v3 (最新、最高精度) [GPU推奨]")
        
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
        if platform.system() == "Linux":
            print("\nUbuntuでの一般的な解決方法:")
            print("1. sudo apt-get update && sudo apt-get install portaudio19-dev")
            print("2. pip install --upgrade sounddevice")
            print("3. pulseaudio --start")
    finally:
        if transcriber_instance:
            transcriber_instance.stop()

if __name__ == "__main__":
    main() 