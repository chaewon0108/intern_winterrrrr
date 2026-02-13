import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def extract_frame_number(filename):
    try:
        name_without_ext = os.path.splitext(filename)[0]
        frame_num = int(name_without_ext.split('_')[-1])
        return frame_num
    except:
        return -1


def plot_emotion_analysis(csv_new, output_path="emotion_analysis.png"):
    """
    하나의 PNG에 3개 그래프 생성:
    
    1. CAGE 감정 변이 그래프 (선)
    2. CAGE Valence & Arousal 그래프 (선)
    3. CAGE 대표 감정 + Score 그래프 (막대 + 선) - 막대: intensity, 선: esti_score

    """
    df_new = pd.read_csv(csv_new)
    
    df_new['frame_num'] = df_new['filename'].apply(extract_frame_number)
    
    df_new = df_new.sort_values('frame_num')
    
    emotion_cols = ['esti_angry', 'esti_disgust', 'esti_fear', 'esti_happy', 
                    'esti_sad', 'esti_surprise', 'esti_neutral']
    

    emotion_name_to_col = {
        'angry': 'esti_angry',
        'anger': 'esti_angry',
        'disgust': 'esti_disgust',
        'fear': 'esti_fear',
        'happy': 'esti_happy',
        'happiness': 'esti_happy',
        'sad': 'esti_sad',
        'sadness': 'esti_sad',
        'surprise': 'esti_surprise',
        'surprised': 'esti_surprise',
        'neutral': 'esti_neutral'
    }
    
    # 노이즈 보정된 감정 > 필터링된 감정 > 원본 감정 순으로 사용
    if 'noise_esti_expression' in df_new.columns:
        expression_col = 'noise_esti_expression'
    elif 'filtered_expression' in df_new.columns:
        expression_col = 'filtered_expression'
    else:
        expression_col = 'esti_expression'

    df_new['dominant_emotion'] = df_new[expression_col].map(
        lambda x: emotion_name_to_col.get(x, 'esti_neutral')
    )

    
    # 감정별 색상 설정 (컬럼명 기준)
    emotion_colors = {
        'esti_angry': '#FF5252',      # 빨강
        'esti_disgust': "#E06DF4",    # 보라
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
    
    # 하나의 figure에 3개 subplot 생성
    fig = plt.figure(figsize=(16, 15))
    
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5, top=0.90)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    legend_handles = [plt.Rectangle((0,0),1,1, color=emotion_colors[col], alpha=0.7, label=emotion_labels[col]) 
                     for col in emotion_cols] #d위에 색깔별로 감정 
    fig.legend(handles=legend_handles, loc='upper center', ncol=7, fontsize=10, 
               framealpha=0.9, bbox_to_anchor=(0.5, 0.98))
    
    # 1. 감정 변이 그래프 (CAGE)    
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
    
    ax1.set_title('Emotion Variation', fontsize=13, fontweight='bold', pad=8)
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
    
    # 2. Valence & Arousal 그래프 (CAGE) - y축 범위 -1.0~1.0 고정
    if 'cage_valence' in df_new.columns and 'cage_arousal' in df_new.columns:
        valence_data = df_new['cage_valence'].copy()
        arousal_data = df_new['cage_arousal'].copy()
        
        valence_data[df_new['is_detected'] == False] = np.nan
        arousal_data[df_new['is_detected'] == False] = np.nan
        
        # y축 범위를 -1.0~1.0으로 고정
        y_min, y_max = -1.0, 1.0
        print(f"  ✓ Valence/Arousal y축 범위: [{y_min:.3f}, {y_max:.3f}] (고정)")
        
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
        print("CSV에 cage_valence 또는 cage_arousal이 없습니다!")
    
    # 3. 대표 감정 + score 그래프
    if len(df_new) > 0:
        
        total_frames = len(df_new)
        marker_size = max(8, min(40, 1500 / total_frames))

        # 막대 그래프 (intensity)
        for idx, row in df_new.iterrows():
            if pd.notna(row['intensity']) and row['is_detected']:
                color = emotion_colors[row['dominant_emotion']]
                ax3.bar(row['frame_num'], row['intensity'],
                      width=0.9,
                      color=color,
                      alpha=0.85,
                      edgecolor='none')

        # 선 그래프 추가 (1등 감정 score)
        # 노이즈 보정된 score > 필터링된 score > 원본 score 순으로 사용
        if 'noise_esti_score' in df_new.columns:
            score_col = 'noise_esti_score'
        elif 'filtered_score' in df_new.columns:
            score_col = 'filtered_score'
        else:
            score_col = 'esti_score'
        score_data = df_new[score_col].copy()
        score_data[df_new['is_detected'] == False] = np.nan

        # 두 번째 y축 생성 (score용)
        ax3_twin = ax3.twinx()

        # 선 그래프 (마커 없이)
        ax3_twin.plot(df_new['frame_num'], score_data,
                     color='black',
                     linewidth=1.5,
                     alpha=0.5,
                     linestyle='-')

        # 노이즈 여부에 따라 마커 다르게 표시
        if 'is_noise' in df_new.columns:
            # 일반 프레임: 검정 동그라미
            normal_mask = (df_new['is_noise'] != 'T') & (df_new['is_detected'] == True)
            ax3_twin.scatter(df_new.loc[normal_mask, 'frame_num'],
                            score_data[normal_mask],
                            marker='o', color='black', s=marker_size, alpha=0.6, zorder=5)

            noise_base = (df_new['is_noise'] == 'T') & (df_new['is_detected'] == True)

            if 'noise_esti_expression' in df_new.columns:
                # 노이즈인데 감정이 바뀐 프레임: 분홍색 별
                changed_mask = noise_base & (df_new['noise_esti_expression'] != df_new['esti_expression'])
                ax3_twin.scatter(df_new.loc[changed_mask, 'frame_num'],
                                score_data[changed_mask],
                                marker='*', color="#FF69B4", s=marker_size*2, alpha=0.9, zorder=7)

                # 노이즈인데 감정이 안 바뀐 프레임: 연두색 별
                unchanged_mask = noise_base & (df_new['noise_esti_expression'] == df_new['esti_expression'])
                ax3_twin.scatter(df_new.loc[unchanged_mask, 'frame_num'],
                                score_data[unchanged_mask],
                                marker='*', color="#00FF00", s=marker_size*2, alpha=0.9, zorder=6)
            else:
                # noise_esti_expression 컬럼이 없으면 기존 방식
                ax3_twin.scatter(df_new.loc[noise_base, 'frame_num'],
                                score_data[noise_base],
                                marker='*', color="#000000", s=marker_size*2, alpha=0.9, zorder=6)
        else:
            # is_noise 컬럼이 없으면 기존 방식
            ax3_twin.scatter(df_new['frame_num'], score_data,
                            marker='o', color='black', s=marker_size, alpha=0.6, zorder=5)
        
        ax3_twin.set_ylabel('Emotion Score (Probability)', fontsize=11)
        ax3_twin.set_ylim(0, 1.0)
        
        ax3.set_title('Dominant Emotion + Score', fontsize=13, fontweight='bold', pad=8)
        ax3.set_xlabel('Frame Num', fontsize=11)
        ax3.set_ylabel('Intensity sqrt(V²+A²)', fontsize=11)
        
        min_frame = df_new['frame_num'].min()
        max_frame = df_new['frame_num'].max()
        ax3.set_xlim(min_frame - 1, max_frame + 1)
        
        # y축 0~1.0 고정
        ax3.set_ylim(0, 1.0)

        # 0.5 빨간선 추가 
        ax3.axhline(y=0.5, color='red', linestyle='-', linewidth=1, alpha=0.7)
        
        major_ticks = np.arange(0, max_frame + 1, 10)
        ax3.set_xticks(major_ticks)
        minor_ticks = np.arange(0, max_frame + 1, 5)
        ax3.set_xticks(minor_ticks, minor=True)
        
        ax3.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8, axis='y')
        ax3.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5, axis='y')
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
    


    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*60)
    print(f"저장 경로: {output_path}")
    
    plt.close() 


if __name__ == "__main__":
    CSV_FOLDER = "/home/technonia/intern/faceinsight/0205/0211/test_with_ear/ear_kmeans_ver2/csv_kmeans_ver2"
    OUTPUT_FOLDER = "/home/technonia/intern/faceinsight/0205/0211/test_with_ear/ear_kmeans_ver2/graph_kmeans_ver2"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    
    if len(csv_files) == 0:
        print(f"⚠️ {CSV_FOLDER}에 CSV 파일이 없습니다!")
    else:
        print(f"\n{'='*60}")
        print(f"총 {len(csv_files)}개의 CSV 파일을 처리합니다.")
        print(f"{'='*60}\n")
        
        for idx, csv_path in enumerate(csv_files, 1):
            filename = os.path.basename(csv_path)
            
            output_filename = filename.replace('.csv', '.png')
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            print(f"\n[{idx}/{len(csv_files)}] 처리 중: {filename}")
            
            try:
                plot_emotion_analysis(csv_path, output_path)
                print(f"✅ 성공: {output_filename}")
            except Exception as e:
                print(f"❌ 오류 발생: {filename}")
                print(f"   에러 메시지: {e}")
        
        print(f"\n{'='*60}")
        print(f"완료! 총 {len(csv_files)}개")
        print(f"저장 위치: {OUTPUT_FOLDER}")
        print(f"{'='*60}")
