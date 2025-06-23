import streamlit as st
import os
import glob
from datetime import datetime
import wave
import numpy as np
import cv2
from PIL import Image
import io
import base64

def get_audio_duration(wav_file):
    """WAVファイルの長さを取得"""
    try:
        with wave.open(wav_file, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        return 0

def audio_to_base64(audio_file):
    """音声ファイルをbase64エンコード"""
    try:
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode()
    except:
        return None

def image_to_base64(image_file):
    """画像ファイルをbase64エンコード"""
    try:
        with open(image_file, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode()
    except:
        return None

def load_dataset_info(data_folder="data"):
    """データセットの情報を読み込み"""
    if not os.path.exists(data_folder):
        return []
    
    dataset_info = []
    
    # タイムスタンプフォルダを取得
    timestamp_folders = sorted(glob.glob(os.path.join(data_folder, "*")), reverse=True)
    
    for folder_path in timestamp_folders:
        if os.path.isdir(folder_path):
            folder_name = os.path.basename(folder_path)
            
            # フォルダ名からタイムスタンプを解析
            try:
                timestamp = datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_str = folder_name
            
            # 音声ファイルと画像ファイルを検索
            audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
            image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
            
            dataset_info.append({
                'folder_path': folder_path,
                'folder_name': folder_name,
                'timestamp': timestamp_str,
                'audio_file': audio_files[0] if audio_files else None,
                'image_file': image_files[0] if image_files else None,
                'audio_duration': get_audio_duration(audio_files[0]) if audio_files else 0
            })
    
    return dataset_info

def main():
    st.set_page_config(
        page_title="マルチモーダルデータセットビューア",
        page_icon="🎤📷",
        layout="wide"
    )
    
    st.title("🎤📷 マルチモーダルデータセットビューア")
    st.markdown("---")
    
    # サイドバー設定
    st.sidebar.header("設定")
    data_folder = st.sidebar.text_input("データフォルダパス", value="data")
    
    # データセット情報を読み込み
    dataset_info = load_dataset_info(data_folder)
    
    if not dataset_info:
        st.error(f"データフォルダ '{data_folder}' が見つからないか、データがありません。")
        st.info("データセット収集システムを実行してデータを取得してください。")
        return
    
    # 統計情報を表示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総データ数", len(dataset_info))
    with col2:
        audio_count = sum(1 for info in dataset_info if info['audio_file'])
        st.metric("音声ファイル数", audio_count)
    with col3:
        image_count = sum(1 for info in dataset_info if info['image_file'])
        st.metric("画像ファイル数", image_count)
    with col4:
        if dataset_info:
            latest_time = dataset_info[0]['timestamp']
            st.metric("最新データ", latest_time)
    
    st.markdown("---")
    
    # フィルタリングオプション
    col1, col2 = st.columns(2)
    with col1:
        show_audio_only = st.checkbox("音声ファイルのみ表示", value=False)
    with col2:
        show_image_only = st.checkbox("画像ファイルのみ表示", value=False)
    
    # 表示件数制限
    max_display = st.slider("表示件数", min_value=10, max_value=100, value=50, step=10)
    
    st.markdown("---")
    
    # データセット一覧を表示
    st.subheader("📊 データセット一覧")
    
    # フィルタリング
    filtered_info = dataset_info
    if show_audio_only:
        filtered_info = [info for info in filtered_info if info['audio_file']]
    if show_image_only:
        filtered_info = [info for info in filtered_info if info['image_file']]
    
    # 表示件数制限
    filtered_info = filtered_info[:max_display]
    
    if not filtered_info:
        st.warning("表示するデータがありません。")
        return
    
    # データを表示
    for i, info in enumerate(filtered_info):
        with st.container():
            st.markdown(f"### 📁 {info['timestamp']} ({info['folder_name']})")
            
            col1, col2 = st.columns(2)
            
            # 画像表示
            with col1:
                if info['image_file']:
                    st.subheader("📷 画像")
                    try:
                        image = Image.open(info['image_file'])
                        st.image(image, caption=f"画像: {os.path.basename(info['image_file'])}", use_container_width=True)
                    except Exception as e:
                        st.error(f"画像読み込みエラー: {e}")
                else:
                    st.warning("画像ファイルがありません")
            
            # 音声表示
            with col2:
                if info['audio_file']:
                    st.subheader("🎤 音声")
                    st.write(f"**ファイル:** {os.path.basename(info['audio_file'])}")
                    st.write(f"**長さ:** {info['audio_duration']:.1f}秒")
                    
                    # 音声プレーヤーを表示
                    try:
                        audio_base64 = audio_to_base64(info['audio_file'])
                        if audio_base64:
                            st.audio(f"data:audio/wav;base64,{audio_base64}", format="audio/wav")
                        else:
                            st.error("音声ファイルの読み込みに失敗しました")
                    except Exception as e:
                        st.error(f"音声再生エラー: {e}")
                else:
                    st.warning("音声ファイルがありません")
            
            # ファイル情報
            st.markdown("**📋 ファイル情報:**")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                if info['image_file']:
                    file_size = os.path.getsize(info['image_file']) / 1024  # KB
                    st.write(f"画像サイズ: {file_size:.1f} KB")
            with col_info2:
                if info['audio_file']:
                    file_size = os.path.getsize(info['audio_file']) / 1024  # KB
                    st.write(f"音声サイズ: {file_size:.1f} KB")
            
            st.markdown("---")
    
    # フッター
    st.markdown("---")
    st.markdown("**💡 使い方:**")
    st.markdown("- サイドバーでデータフォルダパスを変更できます")
    "- フィルタリングオプションで特定のファイルタイプのみ表示できます"
    "- 表示件数スライダーで表示するデータ数を調整できます"
    "- 各データセットの画像と音声を確認できます"

if __name__ == "__main__":
    main() 