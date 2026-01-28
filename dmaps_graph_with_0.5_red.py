import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_frame_number(filename):
    """
    파일명에서 프레임 번호 추출
    예: '19-23736372_open1-1_0000.jpg' -> 0
        '19-23736372_open1-1_0001.jpg' -> 1
    """
    try:
        name_without_ext = os.path.splitext(filename)[0]
        frame_num = int(name_without_ext.split('_')[-1])
        return frame_num
    except:
        return -1

def get_dominant_emotion(row, emotion_cols):
    """
    가장 높은 확률의 감정 찾기
    """
    emotions = {col: row[col] for col in emotion_cols}
    dominant = max(emotions, key=emotions.get)
    return dominant


def plot_emotion_analysis(csv_new, output_path="emotion_analysis.png"):
    """
    하나의 PNG에 5개 그래프 생성:
    
    1. PyFeat (New) 감정 변이 그래프 (선)
    2. CAGE Valence & Arousal 그래프 (선)
    3. PyFeat 대표 감정 그래프 (막대) - 높이: intensity
    4. PyFeat 대표 감정 + Score 그래프 (막대 + 선) - 막대: intensity, 선: esti_score
    5. PyFeat 대표 감정 + Confidence 그래프 (막대 + 선) - 막대: intensity, 선: confidence
    
    Args:
        csv_new: PyFeat new 버전 CSV 경로 (cage_valence, cage_arousal 포함)
        output_path: 저장할 그래프 경로
    """
    print("\n" + "="*60)
    print("감정 분석 그래프 생성 시작 (5개)")
    print("="*60)
    
    # CSV 파일 읽기
    df_new = pd.read_csv(csv_new)
    
    # 프레임 번호 추출
    df_new['frame_num'] = df_new['filename'].apply(extract_frame_number)
    
    # 프레임 번호로 정렬
    df_new = df_new.sort_values('frame_num')
    
    # 감정 컬럼들
    emotion_cols = ['esti_angry', 'esti_disgust', 'esti_fear', 'esti_happy', 
                    'esti_sad', 'esti_surprise', 'esti_neutral']
    
    # 감정별 색상 설정
    emotion_colors = {
        'esti_angry': '#FF5252',      # 빨강
        'esti_disgust': '#AB47BC',    # 보라
        'esti_fear': '#7B1FA2',       # 진한 보라
        'esti_happy': '#FFD54F',      # 노랑
        'esti_sad': '#42A5F5',        # 파랑
        'esti_surprise': '#FFA726',   # 주황
        'esti_neutral': '#BDBDBD'     # 회색
    }
    
    # 감정 라벨
    emotion_labels = {
        'esti_angry': 'Angry',
        'esti_disgust': 'Disgust',
        'esti_fear': 'Fear',
        'esti_happy': 'Happy',
        'esti_sad': 'Sad',
        'esti_surprise': 'Surprise',
        'esti_neutral': 'Neutral'
    }
    
    # ============================================================
    # 하나의 figure에 5개 subplot 생성
    # ============================================================
    fig = plt.figure(figsize=(16, 20))
    
    # GridSpec으로 레이아웃 설정
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.4, top=0.95)  # top 여백 확보
    
    ax1 = fig.add_subplot(gs[0])  # 감정 변이 (PyFeat New)
    ax2 = fig.add_subplot(gs[1])  # Valence & Arousal
    ax3 = fig.add_subplot(gs[2])  # 대표 감정 (막대)
    ax4 = fig.add_subplot(gs[3])  # 대표 감정 + Score
    ax5 = fig.add_subplot(gs[4])  # 대표 감정 + Confidence
    

    legend_handles = [plt.Rectangle((0,0),1,1, color=emotion_colors[col], alpha=0.7, label=emotion_labels[col]) 
                     for col in emotion_cols]
    fig.legend(handles=legend_handles, loc='upper center', ncol=7, fontsize=10, 
               framealpha=0.9, bbox_to_anchor=(0.5, 0.98))
    
    # 1. 감정 변이 그래프 (PyFeat New)
    print("\n[1/5] PyFeat 감정 변이 그래프 생성 중...")
    
    # y축 범위 계산
    all_emotion_data = []
    for emotion in emotion_cols:
        emotion_data = df_new[emotion].copy()
        emotion_data[df_new['is_detected'] == False] = np.nan
        all_emotion_data.append(emotion_data)
    
    all_values = pd.concat(all_emotion_data)
    valid_values = all_values.dropna()
    
    if len(valid_values) > 0:
        data_min = valid_values.min()
        data_max = valid_values.max()
        margin = (data_max - data_min) * 0.01
        shared_y_min = max(0, data_min - margin)
        shared_y_max = min(1, data_max + margin)
        
        if (shared_y_max - shared_y_min) < 0.15:
            center = (shared_y_max + shared_y_min) / 2
            shared_y_min = max(0, center - 0.075)
            shared_y_max = min(1, center + 0.075)
        
        print(f"  ✓ y축 범위: [{shared_y_min:.3f}, {shared_y_max:.3f}]")
    else:
        shared_y_min, shared_y_max = 0, 1
    
    # 감정 변이 그래프 그리기
    for emotion in emotion_cols:
        emotion_data = df_new[emotion].copy()
        emotion_data[df_new['is_detected'] == False] = np.nan
        
        ax1.plot(df_new['frame_num'], emotion_data, 
               color=emotion_colors[emotion],
               linewidth=1.5,
               alpha=0.6)
    
    ax1.set_title('PyFeat (New) - Emotion Variation', fontsize=13, fontweight='bold', pad=8)
    ax1.set_xlabel('Frame Num', fontsize=11)
    ax1.set_ylabel('Emotion score', fontsize=11)
    ax1.set_ylim(shared_y_min, shared_y_max)
    
    if len(df_new) > 0:
        min_frame = df_new['frame_num'].min()
        max_frame = df_new['frame_num'].max()
        ax1.set_xlim(min_frame - 1, max_frame + 1)
        
        major_ticks = np.arange(0, max_frame + 1, 10)
        ax1.set_xticks(major_ticks)
        minor_ticks = np.arange(0, max_frame + 1, 5)
        ax1.set_xticks(minor_ticks, minor=True)
        
        ax1.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8)
        ax1.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
    
    # y축 눈금
    y_range = shared_y_max - shared_y_min
    if y_range < 0.2:
        tick_interval = 0.02
    elif y_range < 0.3:
        tick_interval = 0.05
    elif y_range < 0.6:
        tick_interval = 0.1
    else:
        tick_interval = 0.2
    
    major_yticks = np.arange(
        np.floor(shared_y_min / tick_interval) * tick_interval,
        np.ceil(shared_y_max / tick_interval) * tick_interval + tick_interval,
        tick_interval
    )
    ax1.set_yticks(major_yticks)
    
    # 2. Valence & Arousal 그래프 (CAGE)    
    if 'cage_valence' in df_new.columns and 'cage_arousal' in df_new.columns:
        valence_data = df_new['cage_valence'].copy()
        arousal_data = df_new['cage_arousal'].copy()
        
        valence_data[df_new['is_detected'] == False] = np.nan
        arousal_data[df_new['is_detected'] == False] = np.nan
        
        valence_valid = valence_data.dropna()
        arousal_valid = arousal_data.dropna()
        
        if len(valence_valid) > 0 and len(arousal_valid) > 0:
            all_min = min(valence_valid.min(), arousal_valid.min())
            all_max = max(valence_valid.max(), arousal_valid.max())
            
            margin = (all_max - all_min) * 0.2
            y_min = all_min - margin
            y_max = all_max + margin
            
            if (y_max - y_min) < 0.6:
                center = (y_max + y_min) / 2
                y_min = center - 0.3
                y_max = center + 0.3
            
            y_min = max(y_min, -1.0)
            y_max = min(y_max, 1.0)
            
            print(f"  ✓ Valence/Arousal y축 범위: [{y_min:.3f}, {y_max:.3f}]")
        else:
            y_min, y_max = -1.0, 1.0
        
        ax2.plot(df_new['frame_num'], valence_data,
                color='#4CAF50', linewidth=2, alpha=0.8, label='Valence')
        ax2.plot(df_new['frame_num'], arousal_data,
                color='#F44336', linewidth=2, alpha=0.8, label='Arousal')
        
        if y_min <= 0 <= y_max:
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_title('Valence & Arousal (CAGE)', fontsize=13, fontweight='bold', pad=8)
        ax2.set_xlabel('Frame Num', fontsize=11)
        ax2.set_ylabel('Value', fontsize=11)
        ax2.set_ylim(y_min, y_max)
        
        if len(df_new) > 0:
            min_frame = df_new['frame_num'].min()
            max_frame = df_new['frame_num'].max()
            ax2.set_xlim(min_frame - 1, max_frame + 1)
            
            major_ticks = np.arange(0, max_frame + 1, 10)
            ax2.set_xticks(major_ticks)
            minor_ticks = np.arange(0, max_frame + 1, 5)
            ax2.set_xticks(minor_ticks, minor=True)
            
            ax2.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8)
            ax2.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
            ax2.tick_params(axis='x', rotation=45, labelsize=9)
        
        ax2.legend(loc='upper right', fontsize=10, frameon=True)
    else:
        print("  ⚠️ CSV에 cage_valence 또는 cage_arousal이 없습니다!")
    
    # 3 대표감정그래프    
    # intensity 계산 (sqrt(valence² + arousal²))
    if 'cage_valence' in df_new.columns and 'cage_arousal' in df_new.columns:
        df_new['intensity'] = df_new.apply(
            lambda row: np.sqrt(row['cage_valence']**2 + row['cage_arousal']**2) 
            if row['is_detected'] and pd.notna(row['cage_valence']) and pd.notna(row['cage_arousal']) 
            else np.nan,
            axis=1
        )
        print("  ✓ intensity 계산 완료 (sqrt(valence² + arousal²))")
    else:
        df_new['intensity'] = np.nan
        print("  ⚠️ cage_valence/arousal 없어서 intensity 계산 불가")
    
    # 대표 감정 찾기
    df_new['dominant_emotion'] = df_new.apply(
        lambda row: get_dominant_emotion(row, emotion_cols) if row['is_detected'] else 'esti_neutral',
        axis=1
    )
    
    # esti_score 확인
    if 'esti_score' not in df_new.columns:
        print("  ⚠️ PyFeat CSV에 esti_score가 없습니다!")
    else:
        print("  ✓ PyFeat CSV에서 esti_score 불러옴")
    
    # confidence 확인
    if 'confidence' not in df_new.columns:
        print("  ⚠️ PyFeat CSV에 confidence가 없습니다!")
    else:
        print("  ✓ PyFeat CSV에서 confidence 불러옴")
    
    new_emotions = df_new['dominant_emotion'].value_counts()
    print(f"  ✓ PyFeat New 대표 감정 분포: {dict(new_emotions)}")
    
    # 3. PyFeat 대표 감정 그래프 (막대만)
    print("\n[3/5] PyFeat 대표 감정 그래프 (막대) 생성 중...")
    
    if len(df_new) > 0:
        for idx, row in df_new.iterrows():
            if pd.notna(row['intensity']):
                color = emotion_colors[row['dominant_emotion']]
                ax3.bar(row['frame_num'], row['intensity'], 
                      width=0.9,
                      color=color, 
                      alpha=0.85,
                      edgecolor='none')
        
        ax3.set_title('PyFeat New - Dominant Emotion', fontsize=13, fontweight='bold', pad=8)
        ax3.set_xlabel('Frame Num', fontsize=11)
        ax3.set_ylabel('Intensity sqrt(V²+A²)', fontsize=11)
        
        min_frame = df_new['frame_num'].min()
        max_frame = df_new['frame_num'].max()
        ax3.set_xlim(min_frame - 1, max_frame + 1)
        
        # y축 0~1.0 고정
        ax3.set_ylim(0, 1.0)
        
        major_ticks = np.arange(0, max_frame + 1, 10)
        ax3.set_xticks(major_ticks)
        minor_ticks = np.arange(0, max_frame + 1, 5)
        ax3.set_xticks(minor_ticks, minor=True)
        
        ax3.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8, axis='y')
        ax3.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
    
    # 4. PyFeat 대표 감정 + Score 그래프 (막대 + 선)    
    if len(df_new) > 0:
        # 막대 그래프 (3번과 동일 - intensity)
        for idx, row in df_new.iterrows():
            if pd.notna(row['intensity']):
                color = emotion_colors[row['dominant_emotion']]
                ax4.bar(row['frame_num'], row['intensity'], 
                      width=0.9,
                      color=color, 
                      alpha=0.85,
                      edgecolor='none')
        
        # 선 그래프 추가 (1등 감정 score) - 검정색
        score_data = df_new['esti_score'].copy()
        score_data[df_new['is_detected'] == False] = np.nan
        
        # 두 번째 y축 생성 (score용)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df_new['frame_num'], score_data,
                     color='black',
                     linewidth=2,
                     alpha=0.5,
                     # label='Emotion Score',
                     linestyle='-',
                     marker='o',
                     markersize=2)
        
        ax4_twin.set_ylabel('Emotion Score (Probability)', fontsize=11)
        ax4_twin.set_ylim(0, 1.0)
        # ax4_twin.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        ax4.set_title('PyFeat New - Dominant Emotion + Score', fontsize=13, fontweight='bold', pad=8)
        ax4.set_xlabel('Frame Num', fontsize=11)
        ax4.set_ylabel('Intensity sqrt(V²+A²)', fontsize=11)
        
        min_frame = df_new['frame_num'].min()
        max_frame = df_new['frame_num'].max()
        ax4.set_xlim(min_frame - 1, max_frame + 1)
        
        # y축 0~1.0 고정
        ax4.set_ylim(0, 1.0)

        # 0.5 빨간선 추가 
        ax4.axhline(y=0.5, color='red', linestyle='-', linewidth=1, alpha=0.7)

        
        major_ticks = np.arange(0, max_frame + 1, 10)
        ax4.set_xticks(major_ticks)
        minor_ticks = np.arange(0, max_frame + 1, 5)
        ax4.set_xticks(minor_ticks, minor=True)
        
        ax4.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8, axis='y')
        ax4.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
        ax4.tick_params(axis='x', rotation=45, labelsize=9)
    
    # 5. PyFeat 대표 감정 + Confidence 그래프 => (막대 + 선)    
    if len(df_new) > 0:
        # 막대 그래프 (3번과 동일 - intensity)
        for idx, row in df_new.iterrows():
            if pd.notna(row['intensity']):
                color = emotion_colors[row['dominant_emotion']]
                ax5.bar(row['frame_num'], row['intensity'], 
                      width=0.9,
                      color=color, 
                      alpha=0.85,
                      edgecolor='none')
        
        # 선 그래프 추가 (얼굴 검출 confidence) - 검정색
        confidence_data = df_new['confidence'].copy()
        confidence_data[df_new['is_detected'] == False] = np.nan
        
        # 두 번째 y축 생성 (confidence용)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(df_new['frame_num'], confidence_data,
                     color='black',
                     linewidth=2,
                     alpha=0.5,
                     # label='Face Confidence',
                     linestyle='-',
                     marker='o',
                     markersize=2)
        
        ax5_twin.set_ylabel('Face Detection Confidence', fontsize=11)
        ax5_twin.set_ylim(0, 1.0)
        # ax5_twin.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        ax5.set_title('PyFeat New - Dominant Emotion + Confidence', fontsize=13, fontweight='bold', pad=8)
        ax5.set_xlabel('Frame Num', fontsize=11)
        ax5.set_ylabel('Intensity sqrt(V²+A²)', fontsize=11)
        
        min_frame = df_new['frame_num'].min()
        max_frame = df_new['frame_num'].max()
        ax5.set_xlim(min_frame - 1, max_frame + 1)
        
        # y축 0~1.0 고정
        ax5.set_ylim(0, 1.0)
        
        major_ticks = np.arange(0, max_frame + 1, 10)
        ax5.set_xticks(major_ticks)
        minor_ticks = np.arange(0, max_frame + 1, 5)
        ax5.set_xticks(minor_ticks, minor=True)
        
        ax5.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8, axis='y')
        ax5.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
        ax5.tick_params(axis='x', rotation=45, labelsize=9)
    
    # ============================================================
    # 저장
    # ============================================================
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print("✅ 그래프 생성 완료!")
    print("="*60)
    print(f"저장 경로: {output_path}")
    print(f"총 5개 그래프:")
    print(f"  1. PyFeat 감정 변이 (선)")
    print(f"  2. Valence & Arousal (CAGE)")
    print(f"  3. PyFeat 대표 감정 (막대, 높이=intensity)")
    print(f"  4. PyFeat 대표 감정 + Score (막대=intensity, 선=score)")
    print(f"  5. PyFeat 대표 감정 + Confidence (막대=intensity, 선=confidence)")
    print("="*60)
    
    plt.close()


if __name__ == "__main__":
    CSV_NEW = "/home/technonia/intern/faceinsight/0128/0128_youtube_dmaps_graph/test/test_csv/0128_test_youtube6.csv"
    OUTPUT_PATH = "/home/technonia/intern/faceinsight/0128/0128_youtube_dmaps_graph/test/test_graph/0128_youtube6.png"
    
    plot_emotion_analysis(CSV_NEW, output_path=OUTPUT_PATH)
    
    print("\n완료!")
