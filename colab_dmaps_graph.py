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
        # 확장자 제거
        name_without_ext = os.path.splitext(filename)[0]
        # 마지막 '_' 뒤의 숫자 추출
        frame_num = int(name_without_ext.split('_')[-1])
        return frame_num
    except:
        return -1


def plot_emotion_comparison(csv_modelpt, csv_new, csv_old, output_path="emotion_comparison.png"):
    """
    4개의 그래프 생성:
    1. model.pt (CAGE) 감정 - 동적 y축 (매우 좁게)
    2. PyFeat (New) 감정 - 1번과 동일한 y축
    3. PyFeat (Old) 감정 - 1번과 동일한 y축
    4. Valence & Arousal (CAGE) - 동적 y축 (넓게)
    
    Args:
        csv_modelpt: model.pt (CAGE) 결과 CSV 경로
        csv_new: PyFeat new 버전 CSV 경로
        csv_old: PyFeat old 버전 CSV 경로
        output_path: 저장할 그래프 경로
    """
    # CSV 파일 읽기
    df_modelpt = pd.read_csv(csv_modelpt)
    df_new = pd.read_csv(csv_new)
    df_old = pd.read_csv(csv_old)
    
    # 프레임 번호 추출
    df_modelpt['frame_num'] = df_modelpt['filename'].apply(extract_frame_number)
    df_new['frame_num'] = df_new['filename'].apply(extract_frame_number)
    df_old['frame_num'] = df_old['filename'].apply(extract_frame_number)
    
    # 프레임 번호로 정렬
    df_modelpt = df_modelpt.sort_values('frame_num')
    df_new = df_new.sort_values('frame_num')
    df_old = df_old.sort_values('frame_num')
    
    # 감정 컬럼들
    emotion_cols = ['esti_angry', 'esti_disgust', 'esti_fear', 'esti_happy', 
                    'esti_sad', 'esti_surprise', 'esti_neutral']
    
    # 감정별 색상 설정
    emotion_colors = {
        'esti_angry': '#FF4444',      # 빨강
        'esti_disgust': '#9C27B0',    # 보라
        'esti_fear': '#4A148C',       # 진한 보라
        'esti_happy': '#FFC107',      # 노랑
        'esti_sad': '#2196F3',        # 파랑
        'esti_surprise': '#FF9800',   # 주황
        'esti_neutral': '#9E9E9E'     # 회색
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
    
    # 그래프 생성 (4개 세로로 배치)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # ★ 1번 그래프의 y축 범위를 먼저 계산 ★
    all_emotion_data = []
    for emotion in emotion_cols:
        emotion_data = df_modelpt[emotion].copy()
        emotion_data[df_modelpt['is_detected'] == False] = np.nan
        all_emotion_data.append(emotion_data)
    
    all_values = pd.concat(all_emotion_data)
    valid_values = all_values.dropna()
    
    # model.pt의 y축 범위 계산
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
        
        print(f"\n✅ 공통 y축 범위 (1-3번 그래프): [{shared_y_min:.3f}, {shared_y_max:.3f}]")
        print(f"   - 데이터 범위 (model.pt 기준): [{data_min:.3f}, {data_max:.3f}]")
        print(f"   - y축 범위 크기: {shared_y_max - shared_y_min:.3f}")
    else:
        shared_y_min, shared_y_max = 0, 1
    
    datasets = [
        (df_modelpt, 'model.pt (CAGE)', axes[0]),
        (df_new, 'PyFeat (New)', axes[1]),
        (df_old, 'PyFeat (Old)', axes[2])
    ]
    
    # 범례용 라인 생성
    legend_lines = []
    for emotion in emotion_cols:
        line, = axes[0].plot([], [], label=emotion_labels[emotion], 
                            color=emotion_colors[emotion],
                            linewidth=2, alpha=0.7)
        legend_lines.append(line)
    
    # 전체 그림의 범례를 상단에 배치
    fig.legend(handles=legend_lines, 
              loc='upper center', 
              ncol=7, 
              fontsize=11,
              frameon=True,
              fancybox=True,
              shadow=True,
              bbox_to_anchor=(0.5, 1.0))
    
    # ★ 각 감정 그래프에 데이터 그리기 (1-3번, 모두 동일한 y축) ★
    for df, title, ax in datasets:
        # 각 감정별로 선 그래프 그리기
        for emotion in emotion_cols:
            emotion_data = df[emotion].copy()
            emotion_data[df['is_detected'] == False] = np.nan
            
            ax.plot(df['frame_num'], emotion_data, 
                   color=emotion_colors[emotion],
                   linewidth=1.5,
                   alpha=0.6)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Emotion Probability', fontsize=12)
        
        # ★ 모든 그래프에 동일한 y축 범위 적용 ★
        ax.set_ylim(shared_y_min, shared_y_max)
        
        # x축 범위 및 눈금
        if len(df) > 0:
            min_frame = df['frame_num'].min()
            max_frame = df['frame_num'].max()
            ax.set_xlim(min_frame - 1, max_frame + 1)
            
            major_ticks = np.arange(0, max_frame + 1, 10)
            ax.set_xticks(major_ticks)
            
            minor_ticks = np.arange(0, max_frame + 1, 5)
            ax.set_xticks(minor_ticks, minor=True)
            
            ax.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8)
            ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
            ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # ★ 동일한 y축 눈금 적용 ★
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
        ax.set_yticks(major_yticks)
        
        minor_yticks = np.arange(
            np.floor(shared_y_min / (tick_interval/2)) * (tick_interval/2),
            np.ceil(shared_y_max / (tick_interval/2)) * (tick_interval/2) + (tick_interval/2),
            tick_interval/2
        )
        ax.set_yticks(minor_yticks, minor=True)
    
    # ★ 4번 그래프: Valence & Arousal (범위를 넓게 잡아서 변화를 잘 보이도록) ★
    ax_va = axes[3]
    
    if 'cage_valence' in df_modelpt.columns and 'cage_arousal' in df_modelpt.columns:
        valence_data = df_modelpt['cage_valence'].copy()
        arousal_data = df_modelpt['cage_arousal'].copy()
        
        valence_data[df_modelpt['is_detected'] == False] = np.nan
        arousal_data[df_modelpt['is_detected'] == False] = np.nan
        
        valence_valid = valence_data.dropna()
        arousal_valid = arousal_data.dropna()
        
        if len(valence_valid) > 0 and len(arousal_valid) > 0:
            all_min = min(valence_valid.min(), arousal_valid.min())
            all_max = max(valence_valid.max(), arousal_valid.max())
            
            # ★ 여유 공간을 20%로 크게 줌 (변화를 잘 보기 위해) ★
            margin = (all_max - all_min) * 0.2
            y_min = all_min - margin
            y_max = all_max + margin
            
            # ★ 최소 범위를 0.6으로 크게 설정 ★
            if (y_max - y_min) < 0.6:
                center = (y_max + y_min) / 2
                y_min = center - 0.3
                y_max = center + 0.3
            
            # -1~1 범위를 벗어나지 않도록 제한
            y_min = max(y_min, -1.0)
            y_max = min(y_max, 1.0)
            
            print(f"\n✅ Valence/Arousal 동적 y축 범위: [{y_min:.3f}, {y_max:.3f}]")
            print(f"   - Valence 범위: [{valence_valid.min():.3f}, {valence_valid.max():.3f}]")
            print(f"   - Arousal 범위: [{arousal_valid.min():.3f}, {arousal_valid.max():.3f}]")
            print(f"   - y축 범위 크기: {y_max - y_min:.3f}")
        else:
            y_min, y_max = -1.0, 1.0
        
        ax_va.plot(df_modelpt['frame_num'], valence_data,
                  color='#4CAF50', linewidth=2, alpha=0.8, label='Valence')
        ax_va.plot(df_modelpt['frame_num'], arousal_data,
                  color='#F44336', linewidth=2, alpha=0.8, label='Arousal')
        
        # 0선 표시 (중립선) - 범위 내에 있을 때만
        if y_min <= 0 <= y_max:
            ax_va.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax_va.set_title('Valence & Arousal (CAGE)', fontsize=14, fontweight='bold', pad=10)
        ax_va.set_xlabel('Frame Number', fontsize=12)
        ax_va.set_ylabel('Value', fontsize=12)
        ax_va.set_ylim(y_min, y_max)
        
        if len(df_modelpt) > 0:
            min_frame = df_modelpt['frame_num'].min()
            max_frame = df_modelpt['frame_num'].max()
            ax_va.set_xlim(min_frame - 1, max_frame + 1)
            
            major_ticks = np.arange(0, max_frame + 1, 10)
            ax_va.set_xticks(major_ticks)
            minor_ticks = np.arange(0, max_frame + 1, 5)
            ax_va.set_xticks(minor_ticks, minor=True)
            
            ax_va.grid(which='major', alpha=0.4, linestyle='-', linewidth=0.8)
            ax_va.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
            ax_va.tick_params(axis='x', rotation=45, labelsize=9)
        
        ax_va.legend(loc='upper right', fontsize=11, frameon=True)
        
        # ★ y축 눈금을 넓게 설정 ★
        y_range = y_max - y_min
        if y_range < 0.4:
            tick_interval = 0.1
        elif y_range < 0.8:
            tick_interval = 0.2
        else:
            tick_interval = 0.5
        
        major_yticks = np.arange(
            np.floor(y_min / tick_interval) * tick_interval,
            np.ceil(y_max / tick_interval) * tick_interval + tick_interval,
            tick_interval
        )
        ax_va.set_yticks(major_yticks)
        
        minor_yticks = np.arange(
            np.floor(y_min / (tick_interval/2)) * (tick_interval/2),
            np.ceil(y_max / (tick_interval/2)) * (tick_interval/2) + (tick_interval/2),
            tick_interval/2
        )
        ax_va.set_yticks(minor_yticks, minor=True)
        
        print(f"   - Valence 평균: {valence_valid.mean():.3f}")
        print(f"   - Arousal 평균: {arousal_valid.mean():.3f}")
    else:
        ax_va.text(0.5, 0.5, 'No Valence/Arousal Data Available\n(cage_valence, cage_arousal columns not found)', 
                  ha='center', va='center', fontsize=12, color='red')
        ax_va.set_title('Valence & Arousal (CAGE)', fontsize=14, fontweight='bold', pad=10)
        ax_va.set_xlabel('Frame Number', fontsize=12)
        ax_va.set_ylabel('Value', fontsize=12)
        print("\n⚠️ Valence/Arousal 데이터가 CSV에 없습니다.")
    
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장 완료: {output_path}")
    plt.close()


if __name__ == "__main__":
    """
    19-23736372_open1-1
    19-27362827_open2_1
    19-62525442_ptsd
    26-12345_open1-1

    # 0123
    youtube1
    youtube2
    youtube3
    "0123_dmaps_pyfeat_new_youtube1_sad.csv" 
    """

    CSV_MODELPT = "/home/technonia/intern/faceinsight/0123_dmaps_model_youtube4.csv"
    CSV_NEW = "/home/technonia/intern/faceinsight/0123_dmaps_pyfeat_new_youtube4.csv"
    CSV_OLD = "/home/technonia/intern/faceinsight/0123_dmaps_pyfeat_old_youtube4.csv"
    
    plot_emotion_comparison(CSV_MODELPT, CSV_NEW, CSV_OLD,
                           output_path="0123_youtube4.png") # 여기도 같이 고쳐!!!!!!!
    
    print("\n완료!")
