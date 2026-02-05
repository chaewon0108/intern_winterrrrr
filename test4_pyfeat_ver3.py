import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain

def calculate_emotion_intensity(valence, arousal):
    """
    감정의 세기 계산
    intensity = sqrt(valence² + arousal²)
    """
    return np.sqrt(valence**2 + arousal**2)

def remove_emotion_noise(df, max_noise_len=3):
    """
    감정 시퀀스에서 노이즈를 제거하는 함수 (3단계 처리, 반복)

    논리:
    1. 연속된 동일 감정 블록(run)으로 분할
    2. 기준점: run 길이 > max_noise_len (≥4) → 신뢰 가능한 감정
    3. noise: run 길이 ≤ max_noise_len (≤3) → 노이즈 후보

    ⚠️ Unknown 처리:
       - unknown은 기준점도 noise도 아님 (완전히 무시)
       - 인접성을 끊는 역할 (기준점과 noise 사이에 unknown이 있으면 인접하지 않음)

    [1단계] 분류:
       - 전체 run 계산 후 기준점/noise/unknown으로 분류

    [2단계] 양쪽 기준점 사이에 낀 noise 처리 (직접 인접):
       - noise의 바로 왼쪽 run이 기준점 AND 바로 오른쪽 run이 기준점일 때만 처리
       - 양쪽 기준점 감정이 같다면 → 해당 감정으로 수정
       - 양쪽 기준점 감정이 다르다면 → 각 기준점의 esti_score 평균 비교 → 높은 쪽으로 수정

    [3단계] 한쪽만 기준점에 인접한 noise 처리:
       - noise의 바로 왼쪽 OR 바로 오른쪽 중 하나만 기준점일 때 처리
       - 인접한 기준점의 감정으로 수정

    [반복] 2~3단계에서 변화가 있으면 1단계부터 다시 실행 (변화 없을 때까지)

    Score는 수정된 감정의 해당 프레임 esti 값 사용

    Args:
        df: 감정 분석 결과 DataFrame
        max_noise_len: 노이즈로 간주할 최대 연속 길이 (기본값: 3)

    Returns:
        노이즈가 제거된 DataFrame
    """
    if len(df) == 0:
        return df

    df = df.copy()

    def calculate_runs(emotions):
        """감정 시퀀스를 run으로 분할"""
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

    # emotion_labels(명사형) → 형용사형 매핑
    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    def update_emotion(df, start, end, target_emotion):
        """감정 업데이터 여기서 -> csv 적용"""
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

        # ===== 1단계: 분류 =====
        emotions = df['esti_expression'].tolist()
        runs = calculate_runs(emotions)

        for run in runs:
            if run['emotion'] == 'unknown':
                run['type'] = 'unknown'
            elif run['length'] > max_noise_len:
                run['type'] = 'anchor'
            else:
                run['type'] = 'noise'

        print(f"\n[반복 {iteration}회차]")
        anchor_count = sum(1 for r in runs if r['type'] == 'anchor')
        noise_count = sum(1 for r in runs if r['type'] == 'noise')
        print(f"  1단계 분류: 기준점 {anchor_count}개, noise {noise_count}개")

        # ===== 2단계: [기준점]-[noise]-[기준점] (직접 인접) =====
        step2_count = 0
        for i, run in enumerate(runs):
            if run['type'] != 'noise':
                continue

            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None

            if left and left['type'] == 'anchor' and right and right['type'] == 'anchor':
                if left['emotion'] == right['emotion']:
                    target = left['emotion']
                    print(f"  2단계: {run['emotion']}({run['start']}-{run['end']}) → {target} (양쪽 같음)")
                else:
                    left_avg = df.loc[left['start']:left['end'], 'esti_score'].mean()
                    right_avg = df.loc[right['start']:right['end'], 'esti_score'].mean()
                    target = right['emotion'] if right_avg > left_avg else left['emotion']
                    print(f"  2단계: {run['emotion']}({run['start']}-{run['end']}) → {target} "
                          f"(왼 {left['emotion']} avg={left_avg:.3f}, 오 {right['emotion']} avg={right_avg:.3f})")

                update_emotion(df, run['start'], run['end'], target)
                changed = True
                step2_count += 1

        # ===== 3단계: [기준점]-[noise] 또는 [noise]-[기준점] (한쪽만 인접) =====
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
                print(f"  3단계: {run['emotion']}({run['start']}-{run['end']}) → {left['emotion']} (왼쪽 기준점)")
                changed = True
                step3_count += 1
            elif right_is_anchor and not left_is_anchor:
                update_emotion(df, run['start'], run['end'], right['emotion'])
                print(f"  3단계: {run['emotion']}({run['start']}-{run['end']}) → {right['emotion']} (오른쪽 기준점)")
                changed = True
                step3_count += 1

        print(f"  결과: 2단계 {step2_count}건, 3단계 {step3_count}건 수정")

        if not changed:
            print(f"  변화 없음 → 종료 (총 {iteration}회 반복)")
            break

    return df

def run_folder_analysis_pyfeat(image_base_path, chain, emotion_classes, save_faces=False, output_face_dir=None):
    """
    PyFeat으로 감정 분석 + 검출된 얼굴 저장
    
    Args:
        image_base_path: 이미지 폴더 경로
        chain: ValenceArousalImageChain 객체
        emotion_classes: 감정 클래스 리스트
        save_faces: 검출된 얼굴 저장 여부
        output_face_dir: 얼굴 저장 디렉토리
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    if not os.path.exists(image_base_path):
        print(f"경로 없음: {image_base_path}")
        return pd.DataFrame()

    image_files = [f for f in os.listdir(image_base_path) if os.path.splitext(f)[1].lower() in valid_extensions]
    image_files.sort() 

    print(f"총 {len(image_files)}개의 이미지를 Py-Feat 엔진으로 분석합니다.")
    
    if save_faces and output_face_dir:
        os.makedirs(output_face_dir, exist_ok=True)
        print(f"검출된 얼굴 저장 경로: {output_face_dir}")

    # PyFeat 순서: [angry, disgust, fear, happy, sad, surprise, neutral]
    output_label_map = {
        0: 'esti_angry', 1: 'esti_disgust', 2: 'esti_fear', 3: 'esti_happy',
        4: 'esti_sad', 5: 'esti_surprise', 6: 'esti_neutral'
    }

    # emotion_labels(명사형) → 형용사형 매핑
    emotion_to_adj = {
        'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
        'happiness': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'
    }

    results_data = []
    total_saved_faces = 0

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

            "esti_angry": 0.0, "esti_disgust": 0.0, "esti_fear": 0.0,
            "esti_happy": 0.0, "esti_sad": 0.0, "esti_surprise": 0.0, "esti_neutral": 0.0,
            
            "cage_valence": 0.0,
            "cage_arousal": 0.0,

            "intensity": 0.0,
            "is_detected": False,
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
                                
                                for face_idx, face in enumerate(valid_faces):
                                    x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                                    
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(w, x2)
                                    y2 = min(h, y2)
                                    
                                    face_crop = input_image[y1:y2, x1:x2]
                                    
                                    if face_crop.size > 0:
                                        face_filename = f"{base_filename}_face{face_idx}.jpg"
                                        face_filepath = os.path.join(output_face_dir, face_filename)
                                        cv2.imwrite(face_filepath, face_crop)
                                        total_saved_faces += 1
                            
                            if hasattr(res_obj, 'emotions') and res_obj.emotions is not None:
                                probs = res_obj.emotions

                                if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                                    dom_idx = np.argmax(probs)
                                    dom_score = probs[dom_idx]

                                    # 명사형 → 형용사형 변환
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

        results_data.append(analysis_res)   
    
    if save_faces:
        print(f"\n✅ 총 {total_saved_faces}개의 얼굴이 저장되었습니다.")

    return pd.DataFrame(results_data)

if __name__ == "__main__":

    # 폴더 경로
    ROOT_PATH = "/home/technonia/intern/dmaps_youtube_sample_image/"
    # ROOT_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/4김경진_img/"
    # CSV 저장 폴더
    OUTPUT_CSV_FOLDER = "/home/technonia/intern/faceinsight/0205/pyfeat/pyfeat_csv/"

    # CSV 파일명 접두사
    CSV_PREFIX = "0205_py_new_"

    SAVE_FACES = False
    OUTPUT_FACE_DIR = ""

    os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)

    chain = ValenceArousalImageChain(cage_model_name="model.pt", cage_device="cpu")
    emotion_classes = chain.au_agent.emotion_labels

    # 하위 폴더 목록 가져오기
    if os.path.exists(ROOT_PATH):
        subfolders = [f for f in os.listdir(ROOT_PATH)
                      if os.path.isdir(os.path.join(ROOT_PATH, f))]
        subfolders.sort()

        print(f"\n{'='*60}")
        print(f"총 {len(subfolders)}개의 폴더를 처리합니다.")
        print(f"{'='*60}\n")

        for idx, folder_name in enumerate(subfolders, 1):
            image_folder_path = os.path.join(ROOT_PATH, folder_name)
            output_csv_path = os.path.join(OUTPUT_CSV_FOLDER, f"{CSV_PREFIX}{folder_name}.csv")

            print(f"\n[{idx}/{len(subfolders)}] 처리 중: {folder_name}")
            print("-" * 50)

            try:
                df_final = run_folder_analysis_pyfeat(
                    image_folder_path,
                    chain,
                    emotion_classes,
                    save_faces=SAVE_FACES,
                    output_face_dir=OUTPUT_FACE_DIR
                )

                # 노이즈 제거 적용
                print("\n노이즈 제거 처리 중...")
                df_final = remove_emotion_noise(df_final, max_noise_len=3)

                df_final.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"✅ 완료: {output_csv_path}")

            except Exception as e:
                print(f"❌ 오류 발생: {folder_name}")
                print(f"   에러: {e}")

        print(f"\n{'='*60}")
        print(f"모든 처리 완료! 총 {len(subfolders)}개 폴더")
        print(f"CSV 저장 위치: {OUTPUT_CSV_FOLDER}")
        print(f"{'='*60}")
    else:
        print(f"경로가 존재하지 않습니다: {ROOT_PATH}")

