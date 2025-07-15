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
import pandas as pd
import csv

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

def load_annotations(data_folder="data"):
    """既存のアノテーションを読み込み"""
    annotation_file = os.path.join(data_folder, "annotations.csv")
    annotations = {}
    
    if os.path.exists(annotation_file):
        try:
            # ファイルサイズをチェック
            if os.path.getsize(annotation_file) == 0:
                st.warning("アノテーションファイルが空です。新しいアノテーションから開始します。")
                return annotations
            
            # CSVファイルを読み込み
            df = pd.read_csv(annotation_file)
            
            # 必要なカラムが存在するかチェック
            required_columns = ['timestamp', 'bit1', 'bit2', 'bit3', 'bit4', 'bit_string']
            if not all(col in df.columns for col in required_columns):
                st.error(f"アノテーションファイルの形式が正しくありません。必要なカラム: {required_columns}")
                return annotations
            
            # データが空でないかチェック
            if df.empty:
                st.info("アノテーションファイルにデータがありません。")
                return annotations
            
            # アノテーションデータを読み込み
            for _, row in df.iterrows():
                try:
                    annotations[row['timestamp']] = {
                        'bit1': bool(row['bit1']),
                        'bit2': bool(row['bit2']),
                        'bit3': bool(row['bit3']),
                        'bit4': bool(row['bit4']),
                        'bit_string': str(row['bit_string'])
                    }
                except Exception as e:
                    st.warning(f"行の読み込みエラー: {e}")
                    continue
            
            st.success(f"アノテーションファイルを読み込みました: {len(annotations)}件")
            
        except pd.errors.EmptyDataError:
            st.warning("アノテーションファイルが空です。新しいアノテーションから開始します。")
        except pd.errors.ParserError as e:
            st.error(f"CSVファイルの解析エラー: {e}")
        except Exception as e:
            st.error(f"アノテーションファイルの読み込みエラー: {e}")
    
    return annotations

def save_annotations(annotations, data_folder="data"):
    """アノテーションをCSVファイルに保存"""
    annotation_file = os.path.join(data_folder, "annotations.csv")
    
    try:
        # アノテーションが空の場合
        if not annotations:
            st.warning("保存するアノテーションがありません。")
            return False
        
        # DataFrameを作成
        data = []
        for timestamp, annotation in annotations.items():
            try:
                data.append({
                    'timestamp': str(timestamp),
                    'bit1': int(annotation['bit1']),
                    'bit2': int(annotation['bit2']),
                    'bit3': int(annotation['bit3']),
                    'bit4': int(annotation['bit4']),
                    'bit_string': str(annotation['bit_string'])
                })
            except Exception as e:
                st.warning(f"アノテーション '{timestamp}' の処理エラー: {e}")
                continue
        
        if not data:
            st.error("保存可能なデータがありません。")
            return False
        
        # DataFrameを作成して保存
        df = pd.DataFrame(data)
        
        # データフォルダが存在しない場合は作成
        os.makedirs(data_folder, exist_ok=True)
        
        # CSVファイルに保存
        df.to_csv(annotation_file, index=False, encoding='utf-8')
        
        st.success(f"アノテーションを保存しました: {len(data)}件")
        return True
        
    except Exception as e:
        st.error(f"アノテーション保存エラー: {e}")
        return False

def get_bit_string(bit1, bit2, bit3, bit4):
    """4つのビットからビット文字列を生成"""
    return f"{int(bit1)}{int(bit2)}{int(bit3)}{int(bit4)}"

def main():
    st.set_page_config(
        page_title="マルチモーダルデータセットアノテーター",
        page_icon="🏷️",
        layout="wide"
    )
    
    st.title("🏷️ マルチモーダルデータセットアノテーター")
    st.markdown("---")
    
    # サイドバー設定
    st.sidebar.header("設定")
    data_folder = st.sidebar.text_input("データフォルダパス", value="data")
    
    # アノテーション設定
    st.sidebar.header("アノテーション設定")
    bit1_label = st.sidebar.text_input("ビット1のラベル", value="特徴1")
    bit2_label = st.sidebar.text_input("ビット2のラベル", value="特徴2")
    bit3_label = st.sidebar.text_input("ビット3のラベル", value="特徴3")
    bit4_label = st.sidebar.text_input("ビット4のラベル", value="特徴4")
    
    # データセット情報を読み込み
    dataset_info = load_dataset_info(data_folder)
    
    if not dataset_info:
        st.error(f"データフォルダ '{data_folder}' が見つからないか、データがありません。")
        st.info("データセット収集システムを実行してデータを取得してください。")
        return
    
    # 既存のアノテーションを読み込み
    annotations = load_annotations(data_folder)
    
    # 統計情報を表示
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("総データ数", len(dataset_info))
    with col2:
        audio_count = sum(1 for info in dataset_info if info['audio_file'])
        st.metric("音声ファイル数", audio_count)
    with col3:
        image_count = sum(1 for info in dataset_info if info['image_file'])
        st.metric("画像ファイル数", image_count)
    with col4:
        annotated_count = len(annotations)
        st.metric("アノテーション済み", annotated_count)
    with col5:
        if dataset_info:
            latest_time = dataset_info[0]['timestamp']
            st.metric("最新データ", latest_time)
    
    st.markdown("---")
    
    # フィルタリングオプション
    col1, col2, col3 = st.columns(3)
    with col1:
        show_audio_only = st.checkbox("音声ファイルのみ表示", value=False)
    with col2:
        show_image_only = st.checkbox("画像ファイルのみ表示", value=False)
    with col3:
        show_annotated_only = st.checkbox("アノテーション済みのみ表示", value=False)
    
    # 表示件数制限
    max_display = st.slider("表示件数", min_value=10, max_value=100, value=50, step=10)
    
    st.markdown("---")
    
    # データセット一覧を表示
    st.subheader("📊 データセット一覧（アノテーション付き）")
    
    # フィルタリング
    filtered_info = dataset_info
    if show_audio_only:
        filtered_info = [info for info in filtered_info if info['audio_file']]
    if show_image_only:
        filtered_info = [info for info in filtered_info if info['image_file']]
    if show_annotated_only:
        filtered_info = [info for info in filtered_info if info['folder_name'] in annotations]
    
    # 表示件数制限
    filtered_info = filtered_info[:max_display]
    
    if not filtered_info:
        st.warning("表示するデータがありません。")
        return
    
    # アノテーション変更を追跡
    annotations_changed = False
    
    # データを表示
    for i, info in enumerate(filtered_info):
        with st.container():
            st.markdown(f"### 📁 {info['timestamp']} ({info['folder_name']})")
            
            # 現在のアノテーションを取得（存在しない場合はデフォルト値）
            current_bits = annotations.get(info['folder_name'], {
                'bit1': False, 'bit2': False, 'bit3': False, 'bit4': False
            })
            
            # アノテーション状態表示
            if info['folder_name'] in annotations:
                current_annotation = annotations[info['folder_name']]
                st.success(f"✅ アノテーション済み: {current_annotation['bit_string']}")
            else:
                st.info("📝 未アノテーション")
            
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
            
            # アノテーション入力
            st.subheader("🏷️ アノテーション")
            
            # チェックボックスを配置
            col_bit1, col_bit2, col_bit3, col_bit4 = st.columns(4)
            
            with col_bit1:
                bit1 = st.checkbox(f"{bit1_label}", value=current_bits['bit1'], key=f"bit1_{info['folder_name']}")
            
            with col_bit2:
                bit2 = st.checkbox(f"{bit2_label}", value=current_bits['bit2'], key=f"bit2_{info['folder_name']}")
            
            with col_bit3:
                bit3 = st.checkbox(f"{bit3_label}", value=current_bits['bit3'], key=f"bit3_{info['folder_name']}")
            
            with col_bit4:
                bit4 = st.checkbox(f"{bit4_label}", value=current_bits['bit4'], key=f"bit4_{info['folder_name']}")
            
            # ビット文字列を生成
            bit_string = get_bit_string(bit1, bit2, bit3, bit4)
            
            # ビット文字列を表示
            st.write(f"**ビットパターン:** `{bit_string}`")
            
            # アノテーションが変更されたかチェック
            old_annotation = annotations.get(info['folder_name'], {
                'bit1': False, 'bit2': False, 'bit3': False, 'bit4': False
            })
            
            if (bit1 != old_annotation['bit1'] or 
                bit2 != old_annotation['bit2'] or 
                bit3 != old_annotation['bit3'] or 
                bit4 != old_annotation['bit4']):
                annotations_changed = True
            
            # アノテーションを更新
            annotations[info['folder_name']] = {
                'bit1': bit1,
                'bit2': bit2,
                'bit3': bit3,
                'bit4': bit4,
                'bit_string': bit_string
            }
            
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
    
    # アノテーション保存ボタン
    st.markdown("---")
    
    # 変更があった場合の通知
    if annotations_changed:
        st.info("🔄 アノテーションに変更があります。保存してください。")
    
    # 保存ボタン
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    with col_save2:
        if st.button("💾 アノテーションを保存", type="primary", use_container_width=True):
            if save_annotations(annotations, data_folder):
                st.success("アノテーションを保存しました！")
                st.rerun()
    
    # フッター
    st.markdown("---")
    st.markdown("**💡 使い方:**")
    st.markdown("- サイドバーでデータフォルダパスとビットラベルを設定できます")
    "- フィルタリングオプションで特定のデータのみ表示できます"
    "- 各データセットの4つのチェックボックスでアノテーションを行います"
    "- アノテーションを変更すると「変更があります」の通知が表示されます"
    "- 「アノテーションを保存」ボタンでCSVファイルに記録します"
    "- アノテーション済みデータは緑色で表示されます"

if __name__ == "__main__":
    main() 