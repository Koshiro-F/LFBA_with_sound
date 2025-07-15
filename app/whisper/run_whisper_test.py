#!/usr/bin/env python3
"""
Faster-Whisper音声文字起こしテストスクリプト実行ファイル
"""

import sys
import os

def main():
    print("=== Faster-Whisper 音声文字起こしテストシステム ===")
    print()
    print("使用可能なテストスクリプト:")
    print("  1. whisper_test.py     - 基本的な文字起こしテスト（5秒録音→文字起こし）")
    print("  2. realtime_whisper.py - リアルタイム文字起こし（連続録音・文字起こし）")
    print("  3. mike_test.py        - USBマイクテスト（録音のみ）")
    print()
    
    while True:
        try:
            choice = input("実行するスクリプトを選択してください (1-3, q: 終了): ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("終了します。")
                break
            elif choice == '1':
                print("\n基本的な文字起こしテストを開始します...")
                print("=" * 50)
                try:
                    from whisper_test import main as whisper_test_main
                    whisper_test_main()
                except ImportError as e:
                    print(f"モジュール読み込みエラー: {e}")
                    print("whisper_test.py が同じフォルダにあることを確認してください。")
                except Exception as e:
                    print(f"実行エラー: {e}")
                break
            elif choice == '2':
                print("\nリアルタイム文字起こしを開始します...")
                print("=" * 50)
                try:
                    from realtime_whisper import main as realtime_main
                    realtime_main()
                except ImportError as e:
                    print(f"モジュール読み込みエラー: {e}")
                    print("realtime_whisper.py が同じフォルダにあることを確認してください。")
                except Exception as e:
                    print(f"実行エラー: {e}")
                break
            elif choice == '3':
                print("\nUSBマイクテストを開始します...")
                print("=" * 50)
                try:
                    from mike_test import main as mike_test_main
                    # mike_test.pyはmain関数がないので、直接実行
                    import mike_test
                except ImportError as e:
                    print(f"モジュール読み込みエラー: {e}")
                    print("mike_test.py が同じフォルダにあることを確認してください。")
                except Exception as e:
                    print(f"実行エラー: {e}")
                break
            else:
                print("1, 2, 3, または q を入力してください。")
        except KeyboardInterrupt:
            print("\n\n終了します。")
            break
        except EOFError:
            print("\n\n終了します。")
            break

if __name__ == "__main__":
    # 必要な依存関係をチェック
    missing_deps = []
    
    try:
        import sounddevice
    except ImportError:
        missing_deps.append("sounddevice")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing_deps.append("faster-whisper")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print("❌ 必要な依存関係が不足しています:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print()
        print("以下のコマンドでインストールしてください:")
        print("pip install sounddevice faster-whisper numpy")
        print()
        print("または poetryを使用している場合:")
        print("poetry add sounddevice faster-whisper numpy")
        sys.exit(1)
    
    main() 