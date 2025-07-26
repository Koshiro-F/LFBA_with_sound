# recorder.py

import sounddevice as sd
import numpy as np

def find_microphone():
    """USBマイクを検索し、最適なデバイスIDを返す"""
    devices = sd.query_devices()
    print("利用可能なオーディオデバイス:")
    usb_mics = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  ID {i}: {device['name']}")
            if 'usb' in device['name'].lower() or 'microphone' in device['name'].lower():
                usb_mics.append((i, device['name']))
    if usb_mics:
        print(f"\n-> USBマイク (ID: {usb_mics[0][0]}) を選択しました。")
        return usb_mics[0][0]
    else:
        default_input = sd.default.device[0]
        if default_input != -1:
             print(f"\n-> USBマイクが見つかりません。デフォルト入力 (ID: {default_input}) を使用します。")
             return default_input
    return None

def record_audio(duration, fs, channels, device_id):
    """指定された設定で録音し、Numpy配列として返す"""
    print(f"\n▶ {duration}秒間の録音を開始します...")
    # dtype='float32'にすることで、後の型変換が不要になる
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32', device=device_id)
    sd.wait()
    print("録音完了。")
    # sounddeviceは(N, channels)の配列を返すので、1次元に変換
    return recording.flatten()