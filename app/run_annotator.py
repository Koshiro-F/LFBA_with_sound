#!/usr/bin/env python3
"""
マルチモーダルデータセットアノテーター起動スクリプト (Poetry対応)
"""

import subprocess
import sys
import os

def check_streamlit():
    """Streamlitがインストールされているかチェック"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_dependencies():
    """Poetryで依存関係をインストール"""
    print("Poetryで依存関係をインストールしています...")
    try:
        # プロジェクトルートディレクトリに移動
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # Poetryで依存関係をインストール
        subprocess.check_call(["poetry", "install"], cwd=project_root)
        print("依存関係のインストールが完了しました。")
        return True
    except subprocess.CalledProcessError:
        print("依存関係のインストールに失敗しました。")
        return False
    except FileNotFoundError:
        print("Poetryが見つかりません。Poetryをインストールしてください:")
        print("curl -sSL https://install.python-poetry.org | python3 -")
        return False

def run_annotator():
    """アノテーターを起動"""
    # 現在のディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    annotator_path = os.path.join(current_dir, "dataset_annotator.py")
    
    if not os.path.exists(annotator_path):
        print(f"エラー: {annotator_path} が見つかりません。")
        return
    
    print("マルチモーダルデータセットアノテーターを起動しています...")
    print("ブラウザが自動で開きます。開かない場合は http://localhost:8501 にアクセスしてください。")
    print("停止するには Ctrl+C を押してください。")
    
    try:
        # Poetry環境でStreamlitアプリケーションを起動
        subprocess.run([
            "poetry", "run", "streamlit", "run", annotator_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\nアノテーターを停止しました。")
    except Exception as e:
        print(f"アノテーターの起動に失敗しました: {e}")

def main():
    print("=== マルチモーダルデータセットアノテーター (Poetry版) ===")
    
    # Streamlitのインストール確認
    if not check_streamlit():
        print("Streamlitがインストールされていません。")
        response = input("Poetryで依存関係をインストールしますか？ (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("依存関係が必要です。手動でインストールしてください:")
            print("poetry install")
            return
    
    # アノテーターを起動
    run_annotator()

if __name__ == "__main__":
    main() 