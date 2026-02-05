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
    감정 시퀀스에서 노이즈를 제거하는 함수 (2단계 처리)

    논리:
    1. 연속된 동일 감정 블록(run)으로 분할
    2. Anchor: run 길이 > max_noise_len인 블록 (신뢰 가능한 감정)
    3. Noise Candidate: run 길이 ≤ max_noise_len인 블록 (잠재적 노이즈)

    ⚠️ Unknown 처리:
       - unknown은 노이즈 제거 대상에서 완전히 제외
       - Anchor로 인정하지 않음
       - 다른 감정으로 변환하지 않음 (그대로 유지)

    [1단계] 양쪽 패턴 기반 제거:
       - 연속된 노이즈가 양쪽 같은 감정으로 둘러싸인 경우 그 감정으로 수정
       - 예: N N D N N → N N N N N (D가 양쪽 N으로 수정)
       - unknown은 검사하지 않으며, 양쪽 패턴 매칭에서도 제외

    [2단계] Anchor 기반 제거:
       - 1단계 후 run 재계산 및 Anchor 재식별
       - 첫 Anchor 앞의 노이즈 → 첫 Anchor의 감정으로 수정
       - 두 Anchor 사이의 노이즈 → 앞쪽 Anchor의 감정으로 수정
       - 마지막 Anchor 뒤의 노이즈 → 마지막 Anchor의 감정으로 수정
       - Anchor가 없으면 수정하지 않음
       - unknown은 Anchor로도 노이즈로도 간주하지 않음

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

        # 마지막 run 추가
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
        # 형용사형으로 변환
        adj_emotion = emotion_to_adj.get(target_emotion.lower(), target_emotion.lower())
        esti_col = f"esti_{adj_emotion}"
        for frame_idx in range(start, end + 1):
            df.at[frame_idx, 'esti_expression'] = adj_emotion
            if esti_col in df.columns:
                df.at[frame_idx, 'esti_score'] = df.at[frame_idx, esti_col]
            df.at[frame_idx, 'is_noise'] = "T"

    # ===== 1단계: 양쪽 패턴 기반 제거 =====
    print("\n[1단계] 양쪽 패턴 기반 노이즈 제거")
    emotions = df['esti_expression'].tolist()
    runs = calculate_runs(emotions)

    pattern_removed_count = 0
    changed = True

    # 변화가 없을 때까지 반복
    while changed:
        changed = False

        # 각 노이즈를 개별적으로 검사
        for i in range(len(runs)):
            # unknown은 건너뜀 (무시)
            if runs[i]['emotion'] == 'unknown':
                continue

            # 노이즈인지 확인
            if runs[i]['length'] <= max_noise_len:
                # 양쪽 run 확인
                left_run = runs[i - 1] if i > 0 else None
                right_run = runs[i + 1] if i + 1 < len(runs) else None

                # 양쪽이 unknown이면 매칭 대상에서 제외
                if left_run and left_run['emotion'] == 'unknown':
                    left_run = None
                if right_run and right_run['emotion'] == 'unknown':
                    right_run = None

                # 양쪽이 같은 감정인지 확인
                if left_run and right_run and left_run['emotion'] == right_run['emotion']:
                    target_emotion = left_run['emotion']
                    noise = runs[i]
                    update_emotion(df, noise['start'], noise['end'], target_emotion)
                    pattern_removed_count += 1

                    # 수정 후 재계산하고 처음부터 다시
                    emotions = df['esti_expression'].tolist()
                    runs = calculate_runs(emotions)
                    changed = True
                    break  # for 루프 중단하고 처음부터 다시

    if pattern_removed_count == 0:
        print("  양쪽 패턴 매칭되는 노이즈 없음")

    # ===== 2단계: Anchor 기반 제거 =====
    print("\n[2단계] Anchor 기반 노이즈 제거")
    emotions = df['esti_expression'].tolist()
    runs = calculate_runs(emotions)

    anchors = []
    noise_candidates = []

    for i, run in enumerate(runs):
        # unknown은 Anchor도 노이즈도 아님 (완전히 무시)
        if run['emotion'] == 'unknown':
            continue

        if run['length'] > max_noise_len:
            anchors.append({'index': i, **run})
        else:
            noise_candidates.append({'index': i, **run})

    # Anchor가 없으면 수정하지 않음
    if len(anchors) == 0:
        print("  Anchor가 없어 노이즈 제거를 수행하지 않습니다.")
        return df

    # Noise Candidate 수정
    anchor_removed_count = 0
    for noise in noise_candidates:
        noise_idx = noise['index']
        target_emotion = None

        # 첫 Anchor 앞의 Noise Candidate
        if noise_idx < anchors[0]['index']:
            target_emotion = anchors[0]['emotion']
            print(f"  첫 Anchor 앞 노이즈: {noise['emotion']}({noise['start']}-{noise['end']}) → {target_emotion}")

        # 마지막 Anchor 뒤의 Noise Candidate
        elif noise_idx > anchors[-1]['index']:
            target_emotion = anchors[-1]['emotion']
            print(f"  마지막 Anchor 뒤 노이즈: {noise['emotion']}({noise['start']}-{noise['end']}) → {target_emotion}")

        # 두 Anchor 사이의 Noise Candidate
        else:
            # 앞쪽 Anchor 찾기
            prev_anchor = None
            for anchor in anchors:
                if anchor['index'] < noise_idx:
                    prev_anchor = anchor
                else:
                    break

            if prev_anchor is not None:
                target_emotion = prev_anchor['emotion']
                print(f"  두 Anchor 사이 노이즈: {noise['emotion']}({noise['start']}-{noise['end']}) → {target_emotion}")

        # 감정과 Score 수정
        if target_emotion is not None:
            update_emotion(df, noise['start'], noise['end'], target_emotion)
            anchor_removed_count += 1

    if anchor_removed_count == 0:
        print("  Anchor 기반 제거할 노이즈 없음")

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
    # CSV 저장 폴더
    OUTPUT_CSV_FOLDER = "/home/technonia/intern/faceinsight/0203/pyfeat/pyfeat_csv/"

    # CSV 파일명 접두사
    CSV_PREFIX = "0203_py_new_"

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

