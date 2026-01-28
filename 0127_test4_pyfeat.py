import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain

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

    print(f"총 {len(image_files)}개의 이미지를 Py-Feat 엔진으로만 분석합니다.")
    
    # 얼굴 저장 디렉토리 생성
    if save_faces and output_face_dir:
        os.makedirs(output_face_dir, exist_ok=True)
        print(f"검출된 얼굴 저장 경로: {output_face_dir}")

    output_label_map = { 
        0: 'esti_angry', 1: 'esti_disgust', 2: 'esti_fear', 3: 'esti_happy', 
        4: 'esti_sad', 5: 'esti_surprise', 6: 'esti_neutral'
    }

    results_data = []
    total_saved_faces = 0

    for filename in tqdm(image_files, desc="Analyzing Py-Feat"):
        img_full_path = os.path.join(image_base_path, filename)
        input_image = cv2.imread(img_full_path)

        analysis_res = {
            "filename": filename, 
            "is_detected": False,
            "count_faces": 0,
            
            "confidence": 0.0, 

            "esti_expression": "unknown",  # 대표(1등) 감정
            "esti_score": 0.0,  # 대표 감정의 확률값 (emotion probability)

            "esti_angry": 0.0, "esti_disgust": 0.0, "esti_fear": 0.0,
            "esti_happy": 0.0, "esti_sad": 0.0, "esti_surprise": 0.0, "esti_neutral": 0.0,
            
            "cage_valence": 0.0,
            "cage_arousal": 0.0,
        }

        if input_image is not None:
            h, w = input_image.shape[:2]
            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            _, _, faces, _, emotions, _ = chain.au_agent.run(image_rgb, ndarray_to_list=False)

            if faces is not None and len(faces) > 0:
                actual_faces = faces[0] if len(faces) > 0 else []
                
                valid_faces = []
                if isinstance(actual_faces, (list, np.ndarray)) and len(actual_faces) > 0:
                    for face in actual_faces:
                        if isinstance(face, (list, np.ndarray)) and len(face) >= 5:
                            valid_faces.append(face)
                
                if len(valid_faces) > 0:
                    analysis_res["is_detected"] = True
                    analysis_res["count_faces"] = len(valid_faces)
                    
                    first_face = valid_faces[0]
                    if len(first_face) >= 5:
                        analysis_res["confidence"] = round(float(first_face[4]), 3)
                    
                    # ★ 검출된 얼굴들 저장 ★
                    if save_faces and output_face_dir:
                        base_filename = os.path.splitext(filename)[0]
                        
                        for face_idx, face in enumerate(valid_faces):
                            x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                            
                            # 이미지 범위 체크
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            
                            # 얼굴 영역 잘라내기
                            face_crop = input_image[y1:y2, x1:x2]
                            
                            if face_crop.size > 0:
                                # 파일명: 원본이미지명_face0.jpg, 원본이미지명_face1.jpg, ...
                                face_filename = f"{base_filename}_face{face_idx}.jpg"
                                face_filepath = os.path.join(output_face_dir, face_filename)
                                cv2.imwrite(face_filepath, face_crop)
                                total_saved_faces += 1
                
                    # face[0]을 기준으로 감정 분석
                    if emotions is not None and len(emotions) > 0:
                        probs = emotions
                        while isinstance(probs, (list, np.ndarray)) and len(probs) > 0 and isinstance(probs[0], (list, np.ndarray)):
                            probs = probs[0]
                        
                        if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                            dom_idx = np.argmax(probs)
                            dom_score = probs[dom_idx]  # ★ 1등 감정의 확률값 (score)
                            
                            analysis_res["esti_expression"] = emotion_classes[dom_idx]
                            analysis_res["esti_score"] = round(float(dom_score), 3)  # ★ 감정 score 저장!

                            for idx, key in output_label_map.items():
                                analysis_res[key] = round(float(probs[idx]), 3)
                    
                    # ★★★ CAGE 모델로 valence/arousal 계산 ★★★
                    if chain.use_cage:
                        try:
                            with torch.no_grad():
                                x1, y1, x2, y2, _ = first_face[:5]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # 얼굴 영역 크롭
                                face_crop = input_image[y1:y2, x1:x2]
                                
                                if face_crop.size > 0:
                                    # RGB 변환
                                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                                    
                                    # CAGE 입력 변환
                                    cage_input = chain.cage_transformer(face_crop_rgb)
                                    cage_input = cage_input.unsqueeze(0).to(chain.cage_device)
                                    
                                    # CAGE 모델 실행
                                    cage_output = chain.cage(cage_input)
                                    cage_output_tensor = cage_output.squeeze(0)
                                    cage_output_np = cage_output_tensor.cpu().numpy()
                                    
                                    # valence/arousal 추출 (7, 8번 인덱스)
                                    if len(cage_output_np) >= 9:
                                        cage_valence = float(cage_output_np[7])
                                        cage_arousal = float(cage_output_np[8])
                                        
                                        analysis_res["cage_valence"] = round(cage_valence, 3)
                                        analysis_res["cage_arousal"] = round(cage_arousal, 3)
                        except Exception as e:
                            print(f"  ⚠️ CAGE 처리 오류 ({filename}): {e}")
                            analysis_res["cage_valence"] = 0.0
                            analysis_res["cage_arousal"] = 0.0

        results_data.append(analysis_res)
    
    if save_faces:
        print(f"\n✅ 총 {total_saved_faces}개의 얼굴이 저장되었습니다.")

    return pd.DataFrame(results_data)

if __name__ == "__main__":
    """
    19-23736372_open1-1
    19-27362827_open2_1 #이게 2명으로 잡혀야 돼 
    19-62525442_ptsd
    26-12345_open1-1

    # 0123
    youtube1
    youtube2
    youtube3_sad

    "/home/technonia/intern/dmaps_youtube_sample_image/youtube6"

    /home/technonia/intern/faceinsight/0127_dmaps_sample_video/1심각상진_img/25-00023433_open2_2/

    
    """
    IMAGE_BASE_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/1심각상진_img/25-00023433_open2_2/"
    OUTPUT_CSV_PATH = "/home/technonia/intern/faceinsight/0127_youtube_dmaps_csv/0127_pyfeat_new_심각상진_open2_2.csv" #이제 old 안 할거임! 
    
    # face[i] 저장 파일
    SAVE_FACES = False  # True로 설정하면 얼굴 저장
    OUTPUT_FACE_DIR = ""  # 얼굴 저장 경로

    chain = ValenceArousalImageChain(cage_model_name="model.pt", cage_device="cpu")
    emotion_classes = chain.au_agent.emotion_labels

    if os.path.exists(IMAGE_BASE_PATH):
        df_final = run_folder_analysis_pyfeat(
            IMAGE_BASE_PATH, 
            chain, 
            emotion_classes,
            save_faces=SAVE_FACES,
            output_face_dir=OUTPUT_FACE_DIR
        )
        
        df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print(f"Py-Feat 분석 완료! 저장된 파일: {OUTPUT_CSV_PATH}")

        if SAVE_FACES:
            print(f"\n얼굴 저장 경로: {OUTPUT_FACE_DIR}")
        print("="*50)
    else:
        print(f"경로 오류: {IMAGE_BASE_PATH}")
