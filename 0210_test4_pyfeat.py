import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain
from src.faceinsight.engine.agent.action_code_detector import EYE_AR_THRESH
from imutils import face_utils

# EAR 계산에 사용되는 눈 랜드마크 인덱스
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def calculate_emotion_intensity(valence, arousal):
    return np.sqrt(valence**2 + arousal**2)

def remove_emotion_noise(df, max_noise_len=3):
    if len(df) == 0:
        return df

    df = df.copy()

    def calculate_runs(emotions):
        runs = []
        current_emotion = emotions[0]
        start_idx = 0

        for i in range(1, len(emotions)):
            if emotions[i] != current_emotion:
                runs.append({
                    'emotion': current_emotion,
                    'start': start_idx,
                    'end': i - 1,
                    'length': i - start_idx
                })
                current_emotion = emotions[i]
                start_idx = i

        runs.append({
            'emotion': current_emotion,
            'start': start_idx,
            'end': len(emotions) - 1,
            'length': len(emotions) - start_idx
        })
        return runs

    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    def update_emotion(df, start, end, target_emotion):
        adj_emotion = emotion_to_adj.get(target_emotion.lower(), target_emotion.lower())
        esti_col = f"esti_{adj_emotion}"
        for frame_idx in range(start, end + 1):
            df.at[frame_idx, 'esti_expression'] = adj_emotion
            if esti_col in df.columns:
                df.at[frame_idx, 'esti_score'] = df.at[frame_idx, esti_col]
            df.at[frame_idx, 'is_noise'] = "T"

    iteration = 0
    while True:
        iteration += 1
        changed = False

        # 1단계 noise랑 기준점 설정하기
        emotions = df['esti_expression'].tolist()
        runs = calculate_runs(emotions)

        for run in runs:
            if run['emotion'] == 'unknown':
                run['type'] = 'unknown'
            elif run['length'] > max_noise_len:
                run['type'] = 'anchor'
            else:
                run['type'] = 'noise'

        # print(f"\n[반복 {iteration}회차]")
        anchor_count = sum(1 for r in runs if r['type'] == 'anchor')
        noise_count = sum(1 for r in runs if r['type'] == 'noise')
        # print(f"  1단계 분류: 기준점 {anchor_count}개, noise {noise_count}개")

        # 2단계
        step2_count = 0
        for i, run in enumerate(runs):
            if run['type'] != 'noise':
                continue

            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None

            if left and left['type'] == 'anchor' and right and right['type'] == 'anchor':
                if left['emotion'] == right['emotion']:
                    target = left['emotion']
                    # print(f"  2단계: {run['emotion']}({run['start']}-{run['end']}) → {target} (양쪽 같음)")
                else:
                    left_avg = df.loc[left['start']:left['end'], 'esti_score'].mean()
                    right_avg = df.loc[right['start']:right['end'], 'esti_score'].mean()
                    target = right['emotion'] if right_avg > left_avg else left['emotion']
                    # print(f"  2단계: {run['emotion']}({run['start']}-{run['end']}) → {target} "
                          # f"(왼 {left['emotion']} avg={left_avg:.3f}, 오 {right['emotion']} avg={right_avg:.3f})")

                update_emotion(df, run['start'], run['end'], target)
                changed = True
                step2_count += 1

        # 3단계
        step3_count = 0
        for i, run in enumerate(runs):
            if run['type'] != 'noise':
                continue
            # 2단계에서 이미 처리된 경우 건너뜀
            if df.at[run['start'], 'is_noise'] == 'T':
                continue

            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            left_is_anchor = left is not None and left['type'] == 'anchor'
            right_is_anchor = right is not None and right['type'] == 'anchor'

            if left_is_anchor and not right_is_anchor:
                update_emotion(df, run['start'], run['end'], left['emotion'])
                # print(f"  3단계: {run['emotion']}({run['start']}-{run['end']}) → {left['emotion']} (왼쪽 기준점)")
                changed = True
                step3_count += 1
            elif right_is_anchor and not left_is_anchor:
                update_emotion(df, run['start'], run['end'], right['emotion'])
                # print(f"  3단계: {run['emotion']}({run['start']}-{run['end']}) → {right['emotion']} (오른쪽 기준점)")
                changed = True
                step3_count += 1

        # print(f"  결과: 2단계 {step2_count}건, 3단계 {step3_count}건 수정")

        if not changed:
            # print(f"  변화 없음 → 종료 (총 {iteration}회 반복)")
            break

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

    # PyFeat 순서: angry, disgust, fear, happy, sad, surprise, neutral
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
    total_times = {
        'face_detect': 0.0,
        'landmark': 0.0,
        'ear_calc': 0.0,
        'landmark_ear': 0.0,
        'emotion': 0.0,
        'total': 0.0,
    }

    for filename in tqdm(image_files, desc="Analyzing Py-Feat"):
        img_full_path = os.path.join(image_base_path, filename)
        input_image = cv2.imread(img_full_path)

        analysis_res = {
            "filename": filename, 
            
            "count_faces": 0,
            
            "confidence": 0.0, 

            "esti_expression": "unknown",
            "esti_score": 0.0,
            
            "esti_angry": 0.0, "esti_disgust": 0.0, "esti_fear": 0.0,
            "esti_happy": 0.0, "esti_sad": 0.0, "esti_surprise": 0.0, "esti_neutral": 0.0,

            "cage_valence": 0.0,
            "cage_arousal": 0.0,

            "intensity": 0.0,
            "is_detected": False,

            "ear": 0.0,
            "ear_left": 0.0,
            "ear_right": 0.0,

            "is_noise": "F",

            "eye_open_result": "unknown",
            "ear_avg_result": "unknown",
            "ear_avg_left_result": "unknown",
            "ear_avg_right_result": "unknown",
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
                        valid_faces = []
                        for face in faces:
                            if isinstance(face, (list, np.ndarray)) and len(face) >= 5:
                                valid_faces.append(face)
                        
                        if len(valid_faces) > 0:
                            analysis_res["is_detected"] = True
                            analysis_res["count_faces"] = len(valid_faces)
                            
                            first_face = valid_faces[0]
                            if len(first_face) >= 5:
                                analysis_res["confidence"] = round(float(first_face[4]), 3)
                            
                            # 얼굴 저장
                            if save_faces and output_face_dir:
                                base_filename = os.path.splitext(filename)[0]

                                # landmarks 가져오기
                                landmarks = None
                                if hasattr(res_obj, 'landmarks') and res_obj.landmarks is not None:
                                    if len(res_obj.landmarks) > 0 and len(res_obj.landmarks[0]) > 0:
                                        landmarks = res_obj.landmarks[0][0]

                                for face_idx, face in enumerate(valid_faces):
                                    x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(w, x2)
                                    y2 = min(h, y2)

                                    face_crop = input_image[y1:y2, x1:x2].copy()

                                    if face_crop.size > 0:
                                        # EAR 계산에 사용되는 눈 랜드마크 그리기
                                        if landmarks is not None and face_idx == 0:
                                            leftEye = landmarks[leftEyeStart:leftEyeEnd]
                                            rightEye = landmarks[rightEyeStart:rightEyeEnd]

                                            # 왼쪽 눈 포인트 그리기 (민트)
                                            for i, pt in enumerate(leftEye):
                                                px = int(pt[0]) - x1
                                                py = int(pt[1]) - y1
                                                cv2.circle(face_crop, (px, py), 1, (255, 229, 85), -1)  # #55E5FF 민트
                                                cv2.putText(face_crop, str(i), (px+2, py-2),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 229, 85), 1) #0.1이 폰트 크기

                                            # 오른쪽 눈 포인트 그리기 (핑크)
                                            for i, pt in enumerate(rightEye):
                                                px = int(pt[0]) - x1
                                                py = int(pt[1]) - y1
                                                cv2.circle(face_crop, (px, py), 1, (255, 85, 207), -1)  # #CF55FF 핑크
                                                cv2.putText(face_crop, str(i), (px+2, py-2),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 85, 207), 1)

                                            # EAR 계산 선 그리기
                                            # 왼쪽 눈: A(1-5), B(2-4), C(0-3)
                                            for eye, color in [(leftEye, (85, 193, 255)), (rightEye, (103, 57, 254))]:
                                                pt0 = (int(eye[0][0]) - x1, int(eye[0][1]) - y1)
                                                pt1 = (int(eye[1][0]) - x1, int(eye[1][1]) - y1)
                                                pt2 = (int(eye[2][0]) - x1, int(eye[2][1]) - y1)
                                                pt3 = (int(eye[3][0]) - x1, int(eye[3][1]) - y1)
                                                pt4 = (int(eye[4][0]) - x1, int(eye[4][1]) - y1)
                                                pt5 = (int(eye[5][0]) - x1, int(eye[5][1]) - y1)

                                                # 수직 선 (A: 1-5, B: 2-4)
                                                cv2.line(face_crop, pt1, pt5, color, 1)
                                                cv2.line(face_crop, pt2, pt4, color, 1)
                                                # 수평 선 (C: 0-3)
                                                cv2.line(face_crop, pt0, pt3, color, 1)

                                        face_filename = f"{base_filename}_face{face_idx}.jpg"
                                        face_filepath = os.path.join(output_face_dir, face_filename)
                                        cv2.imwrite(face_filepath, face_crop)
                                        total_saved_faces += 1
                            
                            if hasattr(res_obj, 'emotions') and res_obj.emotions is not None:
                                probs = res_obj.emotions

                                if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                                    dom_idx = np.argmax(probs)
                                    dom_score = probs[dom_idx]

                                    raw_emotion = emotion_classes[dom_idx]
                                    adj_emotion = emotion_to_adj.get(raw_emotion, raw_emotion)
                                    analysis_res["esti_expression"] = adj_emotion
                                    analysis_res["esti_score"] = round(float(dom_score), 3)

                                    for idx, key in output_label_map.items():
                                        analysis_res[key] = round(float(probs[idx]), 3)
                            
                            if hasattr(res_obj, 'cage_valence'):
                                analysis_res["cage_valence"] = round(float(res_obj.cage_valence), 3)
                            if hasattr(res_obj, 'cage_arousal'):
                                analysis_res["cage_arousal"] = round(float(res_obj.cage_arousal), 3)
                            
                            if hasattr(res_obj, 'cage_valence') and hasattr(res_obj, 'cage_arousal'):
                                valence = float(res_obj.cage_valence)
                                arousal = float(res_obj.cage_arousal)
                                intensity = calculate_emotion_intensity(valence, arousal) #함수불러와서 잘 넣고
                                analysis_res["intensity"] = round(intensity, 3) #3자리까지 하고

                            # EAR 값 저장
                            if hasattr(res_obj, 'ear') and res_obj.ear:
                                ear_data = res_obj.ear
                                if isinstance(ear_data, dict):
                                    ear_value = float(ear_data.get('eye', 0.0))
                                    analysis_res["ear"] = round(ear_value, 4)
                                    analysis_res["ear_left"] = round(float(ear_data.get('left_eye', 0.0)), 4)
                                    analysis_res["ear_right"] = round(float(ear_data.get('right_eye', 0.0)), 4)
                                    if ear_value == 0.0 and float(ear_data.get('left_eye', 0.0)) == 0.0 and float(ear_data.get('right_eye', 0.0)) == 0.0:
                                        analysis_res["eye_open_result"] = "unknown"
                                    else:
                                        analysis_res["eye_open_result"] = "O" if ear_value >= EYE_AR_THRESH else "C"

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

    # 유효한 EAR 값(is_detected=True이고 ear > 0)의 평균으로 O/C 판정
    valid_mask = (df['is_detected'] == True) & (df['ear'] > 0)
    valid_ear = df.loc[valid_mask, 'ear']
    valid_ear_left = df.loc[valid_mask, 'ear_left']
    valid_ear_right = df.loc[valid_mask, 'ear_right']
    if len(valid_ear) > 0:
        ear_avg = valid_ear.mean()
        ear_left_avg = valid_ear_left.mean()
        ear_right_avg = valid_ear_right.mean()
        print(f"*** EAR 평균: {ear_avg:.3f}, left 평균: {ear_left_avg:.3f}, right: {ear_right_avg:.3f} (유효 프레임 {len(valid_ear)}개) ***")
        for i, row in df.iterrows():
            if row['is_detected'] and row['ear'] > 0:
                df.at[i, 'ear_avg_result'] = "O" if row['ear'] >= ear_avg else "C"
                df.at[i, 'ear_avg_left_result'] = "O" if row['ear_left'] >= ear_left_avg else "C"
                df.at[i, 'ear_avg_right_result'] = "O" if row['ear_right'] >= ear_right_avg else "C"

    return df, total_times

if __name__ == "__main__":

    ROOT_PATH = "/home/technonia/intern/dmaps_youtube_sample_image/" #youtube
    # ROOT_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/4김경진_img/" #dmaps
    
    OUTPUT_CSV_FOLDER = "/home/technonia/intern/faceinsight/0205/0206/earO/pyfeat_csv_with_ear"

    CSV_PREFIX = "0209_py_new_" #앞에 접두사 설정

    SAVE_FACES = True
    OUTPUT_FACE_DIR = "/home/technonia/intern/faceinsight/0205/0206/earO/face_with_ear/youtube"

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
                df_final = remove_emotion_noise(df_final, max_noise_len=3)

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
            print(f"  [{folder_name}]")
            print(f"    전체 시간:              {times['total']:.2f}초")
            print(f"    ├ 얼굴 감지:            {times['face_detect']:.2f}초")
            print(f"    ├ 랜드마크 감지:         {times['landmark']:.2f}초")
            print(f"    ├ EAR 계산:             {times['ear_calc']:.2f}초")
            print(f"    ├ (랜드마크+EAR 합계):   {times['landmark_ear']:.2f}초")
            print(f"    ├ 감정 감지:             {times['emotion']:.2f}초")
            print(f"    └ 랜드마크+EAR 제외:     {other_time:.2f}초")
        print(f"{'='*60}")
    else:
        print(f"경로 존재 X: {ROOT_PATH}")

