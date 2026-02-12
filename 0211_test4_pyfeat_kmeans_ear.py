import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans  # <--- [추가] K-Means 임포트

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
    
    # 예외 1: 데이터가 너무 적거나(5개 미만), 분산이 거의 없는 경우(0.02 미만)
    # -> 그냥 기본값 0.20 혹은 평균을 기준으로 삼음 (안전 장치)
    if len(data) < 5 or np.std(data) < 0.02:
        return 0.20  

    # K-Means 클러스터링 (2개 그룹)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data)
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    low_center = centers[0]  # 감은 눈 그룹 중심
    high_center = centers[1] # 뜬 눈 그룹 중심
    
    # 예외 2: 두 그룹 간의 차이가 너무 미세한 경우 (0.05 미만)
    # -> 눈을 계속 감고 있었거나, 계속 뜨고 있었던 경우임
    if (high_center - low_center) < 0.05:
        # 전체적으로 값이 낮으면(0.22 미만) 다 감은 것으로 간주 -> 임계값 높임
        if high_center < 0.22:
            return 0.50 
        # 전체적으로 값이 높으면 다 뜬 것으로 간주 -> 임계값 낮춤
        else:
            return 0.15 

    # 정상: 두 그룹 중심의 중간값을 임계값으로 설정
    threshold = (low_center + high_center) / 2
    return threshold
# -------------------------------------------------------

def calculate_emotion_intensity(valence, arousal):
    return np.sqrt(valence**2 + arousal**2)

def remove_emotion_noise(df, min_anchor_len=4, min_c_count=2):
    # (기존 코드와 동일하여 생략, 그대로 사용하시면 됩니다)
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
        # 기존 로직 유지 (필요 시 ear_kmeans를 여기에 포함시킬지 결정 필요, 현재는 기존 컬럼만 사용)
        check_cols = ['ear_avg_result', 'ear_avg_left_result', 'ear_avg_right_result']
        return sum(1 for col in check_cols if df.at[frame_idx, col] == 'C')

    noise_mask = [get_c_count(i) >= min_c_count for i in range(len(df))]

    for i in range(len(df)):
        if noise_mask[i]:
            df.at[i, 'is_noise'] = "T"

    # ... (이하 remove_emotion_noise 로직 동일) ...
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
            "ear": 0.0, "ear_left": 0.0, "ear_right": 0.0,

            "eye_open_result": "unknown",
            "ear_avg_result": "unknown",
            "ear_avg_left_result": "unknown",
            "ear_avg_right_result": "unknown",
            
            "ear_kmeans": "unknown"  # <--- [추가] K-Means 결과 저장을 위한 초기화
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
                            
                            # 얼굴 저장 및 랜드마크 그리기 로직 (생략 - 기존 코드 유지)
                            if save_faces and output_face_dir:
                                # ... (기존 저장 로직 유지) ...
                                pass # (내용은 그대로 두시면 됩니다)
                            
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
                                    
                                    # 기본적인 임계값(0.18 등) 비교 (참고용)
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

    # -------------------------------------------------------------------------
    # [수정] EAR 후처리: 평균(Average) 방식과 K-Means 방식 모두 적용
    # -------------------------------------------------------------------------
    
    # 유효한 EAR 값 추출 (얼굴이 감지되었고 EAR이 0보다 큰 경우)
    valid_mask = (df['is_detected'] == True) & (df['ear'] > 0)
    valid_ear = df.loc[valid_mask, 'ear']
    valid_ear_left = df.loc[valid_mask, 'ear_left']
    valid_ear_right = df.loc[valid_mask, 'ear_right']
    
    if len(valid_ear) > 0:
        # 1. 기존 방식: 단순 평균(Mean) 사용
        ear_avg = valid_ear.mean()
        ear_left_avg = valid_ear_left.mean()
        ear_right_avg = valid_ear_right.mean()
        
        # 2. [추가] 신규 방식: K-Means 클러스터링을 통한 동적 임계값 계산
        ear_kmeans_thresh = get_dynamic_kmeans_threshold(valid_ear.tolist())

        print(f"*** EAR Stats (N={len(valid_ear)}) ***")
        print(f"  - Mean Threshold: {ear_avg:.3f}")
        print(f"  - K-Means Threshold: {ear_kmeans_thresh:.3f}")

        # DataFrame 업데이트
        for i, row in df.iterrows():
            if row['is_detected'] and row['ear'] > 0:
                # 기존 평균 기반 판정
                df.at[i, 'ear_avg_result'] = "O" if row['ear'] >= ear_avg else "C"
                df.at[i, 'ear_avg_left_result'] = "O" if row['ear_left'] >= ear_left_avg else "C"
                df.at[i, 'ear_avg_right_result'] = "O" if row['ear_right'] >= ear_right_avg else "C"
                
                # [추가] K-Means 기반 판정 적용
                df.at[i, 'ear_kmeans'] = "O" if row['ear'] >= ear_kmeans_thresh else "C"

    return df, total_times

if __name__ == "__main__":

    ROOT_PATH = "/home/technonia/intern/dmaps_youtube_sample_image/" #youtube 폴더
    # ROOT_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/4김경진_img/" #dmaps 폴더
    
    OUTPUT_CSV_FOLDER = "/home/technonia/intern/faceinsight/0205/0211/test_with_ear/csv_kmeans/"

    CSV_PREFIX = "0211_py_new_" #앞에 접두사 설정

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

                # 노이즈 제거 적용
                # print("\n노이즈 제거 처리 중...")
                df_final = remove_emotion_noise(df_final, min_anchor_len=4, min_c_count=2)

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
