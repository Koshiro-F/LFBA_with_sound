import sounddevice as sd
import numpy as np
import wave

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

# 録音パラメータ
fs = 44100       # サンプルレート（Hz）
duration = 5     # 録音時間（秒）
channels = 1     # モノラル（ステレオの場合は2）

# USBマイクを検索
mic_device = find_usb_microphone()

if mic_device is not None:
    print(f"\n録音開始...({duration}秒間)")
    print(f"使用デバイス: {sd.query_devices(mic_device)['name']}")
    
    # USBマイクを使用して録音
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16', device=mic_device)
    sd.wait()  # 録音終了まで待機
    print("録音終了")

    # 保存する WAV ファイルの名前
    output_file = "output.wav"

    # wave モジュールを使って WAV ファイルとして保存
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)  # チャネル数
        wf.setsampwidth(2)         # 16bit → 2バイト
        wf.setframerate(fs)        # サンプルレート
        wf.writeframes(recording.tobytes())  # numpy 配列からバイト列に変換して書き込み

    print(f"音声を {output_file} に保存しました。")
else:
    print("録音を中止しました。USBマイクが見つかりません。")