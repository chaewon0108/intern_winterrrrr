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

            "esti_expression": "unknown",
            "esti_score": 0.0,

            "esti_angry": 0.0, "esti_disgust": 0.0, "esti_fear": 0.0,
            "esti_happy": 0.0, "esti_sad": 0.0, "esti_surprise": 0.0, "esti_neutral": 0.0,
            
            "cage_valence": 0.0,
            "cage_arousal": 0.0,

            "intensity": 0.0
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
                                    
                                    analysis_res["esti_expression"] = emotion_classes[dom_idx]
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
    IMAGE_BASE_PATH = "/home/technonia/intern/faceinsight/0127/0127_dmaps_sample_video_img/4김경진_img/12-34567_open1-2/"
    OUTPUT_CSV_PATH = "/home/technonia/intern/faceinsight/0128/pyfeat/pyfeat_csv/0128_pyfeat_new_김경진_open1-2.csv"
    SAVE_FACES = True
    OUTPUT_FACE_DIR = "/home/technonia/intern/faceinsight/0128/pyfeat/py_faces/김경진/김경진_open1-2/"

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
        print(f"완료!{OUTPUT_CSV_PATH}")

        if SAVE_FACES:
            print(f"\n얼굴 저장 경로: {OUTPUT_FACE_DIR}")
        print("="*50)
