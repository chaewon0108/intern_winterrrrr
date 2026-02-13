import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans # kmeans 추가

from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain
from src.faceinsight.engine.agent.action_code_detector import EYE_AR_THRESH
from imutils import face_utils

# EAR 계산에 사용되는 눈 랜드마크 인덱스
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- [추가] K-Means를 이용한 동적 임계값 계산 함수 ---
def get_dynamic_kmeans_threshold(ear_values):
    """
    EAR 값들의 분포를 K-Means(k=2)로 분석하여
    '눈 감음(Low)' 그룹과 '눈 뜸(High)' 그룹 사이의 임계값을 반환합니다.
    """
    data = np.array(ear_values).reshape(-1, 1)
    
    # kmeans 처리
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data)
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    low_center = centers[0]  # eye close (작은값)
    high_center = centers[1] # eye open
    
    # threshold 설정
    threshold = (low_center + high_center) / 2
    return threshold
# -------------------------------------------------------

def calculate_emotion_intensity(valence, arousal):
    return np.sqrt(valence**2 + arousal**2)

def remove_emotion_noise(df, min_anchor_len=4, min_c_count=2):
    # ear의 평균값 사용
    if len(df) == 0:
        return df

    df = df.copy()
    df['noise_esti_expression'] = df['esti_expression'].copy()
    df['noise_esti_score'] = df['esti_score'].copy()

    ear_cols = ['ear_avg_result', 'ear_avg_left_result', 'ear_avg_right_result', 'ear_kmeans'] # <-- ear_kmeans도 노이즈 판단에 참고하고 싶으면 추가 가능

    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    def get_c_count(frame_idx):
        check_cols = ['ear_avg_result', 'ear_avg_left_result', 'ear_avg_right_result']
        return sum(1 for col in check_cols if df.at[frame_idx, col] == 'C')

    noise_mask = [get_c_count(i) >= min_c_count for i in range(len(df))]

    for i in range(len(df)):
        if noise_mask[i]:
            df.at[i, 'is_noise'] = "T"

    runs = []
    current_emotion = None
    start_idx = None

    for i in range(len(df)):
        if noise_mask[i] or df.at[i, 'esti_expression'] == 'unknown':
            if current_emotion is not None:
                runs.append({'emotion': current_emotion, 'start': start_idx, 'end': i - 1, 'length': i - start_idx})
                current_emotion = None
                start_idx = None
        else:
            emotion = df.at[i, 'esti_expression']
            if emotion != current_emotion:
                if current_emotion is not None:
                    runs.append({'emotion': current_emotion, 'start': start_idx, 'end': i - 1, 'length': i - start_idx})
                current_emotion = emotion
                start_idx = i

    if current_emotion is not None:
        runs.append({'emotion': current_emotion, 'start': start_idx, 'end': len(df) - 1, 'length': len(df) - start_idx})

    anchors = [r for r in runs if r['length'] >= min_anchor_len]

    noise_blocks = []
    block_start = None
    for i in range(len(df)):
        if noise_mask[i]:
            if block_start is None: block_start = i
        else:
            if block_start is not None:
                noise_blocks.append((block_start, i - 1))
                block_start = None
    if block_start is not None: noise_blocks.append((block_start, len(df) - 1))

    anchor_by_end = {a['end']: a for a in anchors}
    anchor_by_start = {a['start']: a for a in anchors}

    for block_start, block_end in noise_blocks:
        left_anchor = anchor_by_end.get(block_start - 1)
        right_anchor = anchor_by_start.get(block_end + 1)

        target = None
        if left_anchor and right_anchor:
            if left_anchor['emotion'] == right_anchor['emotion']:
                target = left_anchor['emotion']
            else:
                left_avg = df.loc[left_anchor['start']:left_anchor['end'], 'noise_esti_score'].mean()
                right_avg = df.loc[right_anchor['start']:right_anchor['end'], 'noise_esti_score'].mean()
                target = right_anchor['emotion'] if right_avg > left_avg else left_anchor['emotion']
        elif left_anchor:
            target = left_anchor['emotion']
        elif right_anchor:
            target = right_anchor['emotion']

        if target:
            adj_emotion = emotion_to_adj.get(target.lower(), target.lower())
            esti_col = f"esti_{adj_emotion}"
            for i in range(block_start, block_end + 1):
                df.at[i, 'noise_esti_expression'] = adj_emotion
                if esti_col in df.columns:
                    df.at[i, 'noise_esti_score'] = df.at[i, esti_col]

    return df

def remove_emotion_noise_v2(df, min_anchor_len=4, max_noise_len=3,
                             max_eye_close_ratio=0.25, ear_col='ear_kmeans',
                             max_unknown_gap=10):
    """
    ear을 사용한 noise 처리 (v3)

    NOISE: 길이 <= max_noise_len AND eye_close 포함 → 인접 ANCHOR 감정으로 대체
    ANCHOR: 길이 >= min_anchor_len AND eye_close 비율 < max_eye_close_ratio → 기준점
    KEEP: 나머지 → 원래 감정 유지

    unknown 처리:
    - unknown은 빈 공간으로 두고, 모든 계산에서 제외
    - unknown 연속 <= max_unknown_gap(10)개: 앞뒤 같은 감정이면 이어붙여서 하나의 run으로 처리
    - unknown 연속 > max_unknown_gap(10)개: run을 끊는 경계
    """
    if len(df) == 0:
        return df

    df = df.copy()

    if ear_col not in df.columns:
        for fallback in ['ear_avg_result', 'ear_avg_left_result']:
            if fallback in df.columns:
                ear_col = fallback
                break

    df['noise_esti_expression'] = df['esti_expression'].copy()
    df['noise_esti_score'] = df['esti_score'].copy()
    df['is_noise'] = 'F'
    # is_detected = False면 unknown으로 수정
    df.loc[df['is_detected'] == False, 'is_noise'] = 'unknown'
    df.loc[df['is_detected'] == False, 'noise_esti_expression'] = 'unknown'

    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    # Step 1: 감정 기준 run 구성 (unknown은 일단 경계로 끊기)
    raw_runs = []
    current_emotion = None
    run_start = None
    eye_close_count = 0

    for i in range(len(df)):
        expr = df.at[i, 'esti_expression']

        if expr == 'unknown':
            if current_emotion is not None:
                length = i - run_start
                raw_runs.append({
                    'emotion': current_emotion, 'start': run_start,
                    'end': i - 1, 'length': length,
                    'eye_close_count': eye_close_count,
                    'eye_close_ratio': eye_close_count / length,
                })
                current_emotion = None
                run_start = None
                eye_close_count = 0
        else:
            if expr != current_emotion:
                if current_emotion is not None:
                    length = i - run_start
                    raw_runs.append({
                        'emotion': current_emotion, 'start': run_start,
                        'end': i - 1, 'length': length,
                        'eye_close_count': eye_close_count,
                        'eye_close_ratio': eye_close_count / length,
                    })
                current_emotion = expr
                run_start = i
                eye_close_count = 0

            if ear_col in df.columns and df.at[i, ear_col] == 'C':
                eye_close_count += 1

    if current_emotion is not None:
        length = len(df) - run_start
        raw_runs.append({
            'emotion': current_emotion, 'start': run_start,
            'end': len(df) - 1, 'length': length,
            'eye_close_count': eye_close_count,
            'eye_close_ratio': eye_close_count / length,
        })

    # Step 1.5: unknown 갭 병합
    # unknown 연속 <= max_unknown_gap이고, 앞뒤 감정이 같으면 하나의 run으로 이어붙임
    # (unknown 프레임 자체는 length/eye_close 계산에 포함하지 않음)
    runs = []
    for run in raw_runs:
        if runs:
            prev = runs[-1]
            unknown_gap = run['start'] - prev['end'] - 1
            if (unknown_gap > 0 and unknown_gap <= max_unknown_gap
                    and run['emotion'] == prev['emotion']):
                # 같은 감정 + unknown 갭이 10개 이하 → 병합
                prev['end'] = run['end']
                prev['length'] += run['length']  # unknown 프레임은 포함하지 않음
                prev['eye_close_count'] += run['eye_close_count']
                prev['eye_close_ratio'] = prev['eye_close_count'] / prev['length']
                continue
        runs.append(run.copy())

    # Step 2: run 분류 (NOISE / ANCHOR / KEEP)
    for run in runs:
        if run['length'] <= max_noise_len and run['eye_close_count'] > 0:
            run['run_type'] = 'NOISE'
        elif run['length'] >= min_anchor_len and run['eye_close_ratio'] < max_eye_close_ratio:
            run['run_type'] = 'ANCHOR'
        else:
            run['run_type'] = 'KEEP'

    # Step 3: is_noise 마킹 (unknown 프레임은 건너뜀)
    for run in runs:
        if run['run_type'] == 'NOISE':
            for i in range(run['start'], run['end'] + 1):
                if df.at[i, 'esti_expression'] != 'unknown':
                    df.at[i, 'is_noise'] = 'T'

    # Step 4: NOISE run 대체 (인접 ANCHOR만 사용, unknown 프레임은 건너뜀)
    for pos, noise_run in enumerate(runs):
        if noise_run['run_type'] != 'NOISE':
            continue

        # 바로 왼쪽/오른쪽 run이 ANCHOR인지 확인 (기준점과는 붙어있어야만 인정)
        left_anchor = None
        if pos > 0 and runs[pos - 1]['run_type'] == 'ANCHOR':
            left_anchor = runs[pos - 1]

        right_anchor = None
        if pos < len(runs) - 1 and runs[pos + 1]['run_type'] == 'ANCHOR':
            right_anchor = runs[pos + 1]

        # 대체할 감정 결정
        target_emotion = None
        if left_anchor and right_anchor:
            if left_anchor['emotion'] == right_anchor['emotion']:
                target_emotion = left_anchor['emotion']
            else:
                left_avg = df.loc[left_anchor['start']:left_anchor['end'], 'noise_esti_score'].mean()
                right_avg = df.loc[right_anchor['start']:right_anchor['end'], 'noise_esti_score'].mean()
                target_emotion = right_anchor['emotion'] if right_avg > left_avg else left_anchor['emotion']
        elif left_anchor:
            target_emotion = left_anchor['emotion']
        elif right_anchor:
            target_emotion = right_anchor['emotion']

        # 감정 대체 적용 (unknown 프레임은 건너뜀)
        if target_emotion is not None:
            adj_emotion = emotion_to_adj.get(target_emotion.lower(), target_emotion.lower())
            esti_col = f'esti_{adj_emotion}'
            for i in range(noise_run['start'], noise_run['end'] + 1):
                if df.at[i, 'esti_expression'] == 'unknown':
                    continue
                df.at[i, 'noise_esti_expression'] = adj_emotion
                if esti_col in df.columns:
                    df.at[i, 'noise_esti_score'] = df.at[i, esti_col]

    return df

def run_folder_analysis_pyfeat(image_base_path, chain, emotion_classes, save_faces=False, output_face_dir=None):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.exists(image_base_path):
        print(f"경로 없음: {image_base_path}")
        return pd.DataFrame()

    image_files = [f for f in os.listdir(image_base_path) if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort() 

    print(f"총 {len(image_files)}개의 이미지")
    
    if save_faces and output_face_dir:
        os.makedirs(output_face_dir, exist_ok=True)
        print(f"검출된 얼굴 저장 경로: {output_face_dir}")

    output_label_map = {
        0: 'esti_angry', 1: 'esti_disgust', 2: 'esti_fear', 3: 'esti_happy',
        4: 'esti_sad', 5: 'esti_surprise', 6: 'esti_neutral'
    }

    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    results_data = []
    total_saved_faces = 0
    total_times = {'face_detect': 0.0, 'landmark': 0.0, 'ear_calc': 0.0, 'landmark_ear': 0.0, 'emotion': 0.0, 'total': 0.0}

    for filename in tqdm(image_files, desc="Analyzing Py-Feat"):
        img_full_path = os.path.join(image_base_path, filename)
        input_image = cv2.imread(img_full_path)

        analysis_res = {
            "filename": filename, 
            "count_faces": 0,
            "confidence": 0.0, 

            "esti_expression": "unknown",
            "esti_score": 0.0,

            "is_noise": "F",
            "noise_esti_expression": "unknown",
            "noise_esti_score": 0.0,

            "esti_angry": 0.0, "esti_disgust": 0.0, "esti_fear": 0.0,
            "esti_happy": 0.0, "esti_sad": 0.0, "esti_surprise": 0.0, "esti_neutral": 0.0,
            "cage_valence": 0.0, "cage_arousal": 0.0, "intensity": 0.0,

            "is_detected": False,

            "ear": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,

            "eye_open_result": "unknown",
            "ear_avg_result": "unknown",
            "ear_avg_left_result": "unknown",
            "ear_avg_right_result": "unknown",
            
            "ear_kmeans": "unknown",
        }

        if input_image is not None:
            h, w = input_image.shape[:2]
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            chain.set_frame(image_rgb)
            result_list = chain.run()
            
            if result_list and result_list[0] is not None:
                res_obj = result_list[0]
                
                if hasattr(res_obj, 'faces') and res_obj.faces is not None:
                    faces = res_obj.faces
                    
                    if isinstance(faces, list) and len(faces) > 0:
                        valid_faces = [f for f in faces if isinstance(f, (list, np.ndarray)) and len(f) >= 5]
                        
                        if len(valid_faces) > 0:
                            analysis_res["is_detected"] = True
                            analysis_res["count_faces"] = len(valid_faces)
                            
                            first_face = valid_faces[0]
                            analysis_res["confidence"] = round(float(first_face[4]), 3)
                            
                            # 얼굴 저장, 랜드마크 그리기
                            if save_faces and output_face_dir:
                                pass # 이제 안 할거니까 걍 pass
                            
                            # 감정 확률 처리
                            if hasattr(res_obj, 'emotions') and res_obj.emotions is not None:
                                probs = res_obj.emotions
                                if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                                    dom_idx = np.argmax(probs)
                                    raw_emotion = emotion_classes[dom_idx]
                                    analysis_res["esti_expression"] = emotion_to_adj.get(raw_emotion, raw_emotion)
                                    analysis_res["esti_score"] = round(float(probs[dom_idx]), 3)
                                    for idx, key in output_label_map.items():
                                        analysis_res[key] = round(float(probs[idx]), 3)
                            
                            if hasattr(res_obj, 'cage_valence') and hasattr(res_obj, 'cage_arousal'):
                                analysis_res["cage_valence"] = round(float(res_obj.cage_valence), 3)
                                analysis_res["cage_arousal"] = round(float(res_obj.cage_arousal), 3)
                                analysis_res["intensity"] = round(calculate_emotion_intensity(float(res_obj.cage_valence), float(res_obj.cage_arousal)), 3)

                            # EAR 값 저장
                            if hasattr(res_obj, 'ear') and res_obj.ear:
                                ear_data = res_obj.ear
                                if isinstance(ear_data, dict):
                                    ear_value = float(ear_data.get('eye', 0.0))
                                    analysis_res["ear"] = round(ear_value, 4)
                                    analysis_res["ear_left"] = round(float(ear_data.get('left_eye', 0.0)), 4)
                                    analysis_res["ear_right"] = round(float(ear_data.get('right_eye', 0.0)), 4)
                                    
                                    if ear_value == 0.0:
                                        analysis_res["eye_open_result"] = "unknown"
                                    else:
                                        analysis_res["eye_open_result"] = "O" if ear_value >= EYE_AR_THRESH else "C"

                                    # 시간 기록
                                    total_times['face_detect'] += float(ear_data.get('face_detect_time', 0.0))
                                    total_times['landmark'] += float(ear_data.get('landmark_time', 0.0))
                                    total_times['ear_calc'] += float(ear_data.get('ear_calc_time', 0.0))
                                    total_times['landmark_ear'] += float(ear_data.get('landmark_ear_time', 0.0))
                                    total_times['emotion'] += float(ear_data.get('emotion_time', 0.0))
                                    total_times['total'] += float(ear_data.get('total_time', 0.0))

        results_data.append(analysis_res)   
    
    if save_faces:
        print(f"\n ----> 총 {total_saved_faces}개의 얼굴이 저장되었습니다.")

    df = pd.DataFrame(results_data)

    valid_mask = (df['is_detected'] == True) & (df['ear'] > 0)
    valid_ear = df.loc[valid_mask, 'ear']
    valid_ear_left = df.loc[valid_mask, 'ear_left']
    valid_ear_right = df.loc[valid_mask, 'ear_right']
    
    if len(valid_ear) > 0:
        ear_avg = valid_ear.mean()
        ear_left_avg = valid_ear_left.mean()
        ear_right_avg = valid_ear_right.mean()
        
        ear_kmeans_thresh = get_dynamic_kmeans_threshold(valid_ear.tolist())

        for i, row in df.iterrows():
            if row['is_detected'] and row['ear'] > 0:
                df.at[i, 'ear_avg_result'] = "O" if row['ear'] >= ear_avg else "C"
                df.at[i, 'ear_avg_left_result'] = "O" if row['ear_left'] >= ear_left_avg else "C"
                df.at[i, 'ear_avg_right_result'] = "O" if row['ear_right'] >= ear_right_avg else "C"
                
                df.at[i, 'ear_kmeans'] = "O" if row['ear'] >= ear_kmeans_thresh else "C"

    return df, total_times

if __name__ == "__main__":

    # ROOT_PATH = "/home/technonia/intern/dmaps_youtube_sample_image/" #youtube 폴더
    ROOT_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/4김경진_img/" #dmaps 폴더
    
    OUTPUT_CSV_FOLDER = "/home/technonia/intern/faceinsight/0205/0211/test_with_ear/ear_kmeans_ver2/ver2_csv/"

    CSV_PREFIX = "0212_py_new_4김경진_" #앞에 접두사 설정

    SAVE_FACES = False
    OUTPUT_FACE_DIR = "/home/technonia/intern/faceinsight/0205/0211/face_with_ear/4김경진"

    os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)

    chain = ValenceArousalImageChain(cage_model_name="model.pt", cage_device="cpu")
    emotion_classes = chain.au_agent.emotion_labels

    if os.path.exists(ROOT_PATH):
        subfolders = [f for f in os.listdir(ROOT_PATH)
                      if os.path.isdir(os.path.join(ROOT_PATH, f))]
        subfolders.sort()

        print(f"\n{'='*60}")
        print(f"총 {len(subfolders)}개의 폴더를 처리합니다.")
        print(f"{'='*60}\n")

        time_results = {}

        for idx, folder_name in enumerate(subfolders, 1):
            image_folder_path = os.path.join(ROOT_PATH, folder_name)
            output_csv_path = os.path.join(OUTPUT_CSV_FOLDER, f"{CSV_PREFIX}{folder_name}.csv")

            print(f"\n[{idx}/{len(subfolders)}] 처리 중: {folder_name}")
            print("-" * 50)

            try:
                # 폴더별로 얼굴 저장 경로 생성
                folder_face_dir = os.path.join(OUTPUT_FACE_DIR, folder_name) if SAVE_FACES else None

                df_final, folder_times = run_folder_analysis_pyfeat(
                    image_folder_path,
                    chain,
                    emotion_classes,
                    save_faces=SAVE_FACES,
                    output_face_dir=folder_face_dir
                )
                time_results[folder_name] = folder_times

                df_final = remove_emotion_noise_v2(df_final) #수정~

                df_final.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"완료: {output_csv_path}")

            except Exception as e:
                print(f"   에러: {e}")

        print(f"\n{'='*60}")
        print(f"d완료~ 총 {len(subfolders)}개 폴더")
        print(f"CSV 저장 위치: {OUTPUT_CSV_FOLDER}")

        print(f"\n{'='*60}")
        print("폴더별 구간별 소요 시간:")
        print("-" * 40)
        for folder_name, times in time_results.items():
            other_time = times['total'] - times['landmark_ear']
            # print(f"  [{folder_name}]")
            # print(f"    전체 시간:              {times['total']:.2f}초")
            # print(f"    ├ 얼굴 감지:            {times['face_detect']:.2f}초")
            # print(f"    ├ 랜드마크 감지:         {times['landmark']:.2f}초")
            # print(f"    ├ EAR 계산:             {times['ear_calc']:.2f}초")
            # print(f"    ├ (랜드마크+EAR 합계):   {times['landmark_ear']:.2f}초")
            # print(f"    ├ 감정 감지:             {times['emotion']:.2f}초")
            # print(f"    └ 랜드마크+EAR 제외:     {other_time:.2f}초")
        print(f"{'='*60}")
    else:
        print(f"경로 존재 X: {ROOT_PATH}")
