# model 변경
# /home/technonia/intern/faceinsight/venv/lib/python3.11/site-packages/feat/resources/resmasking_dropout1_rot30_2019Nov17_14.33)
# github 원본 model은 /densenet121_rot30_2019Nov11_14.23
# csv 저장 방법 변경
# affectnet / validation.csv 이용
# csv 만들기!

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain
from src.faceinsight.engine.postprocessor.emotion_calibration_utils import get_dominant_emotion #emotion, expression
import torch
import types # 함수 교체를 위해 필요

# -------- label csv 수정하기 --------------
# face width, face height가 있어서 이걸 x1, x2, y1, y2로 바꾸려고 코드 실행행
# df = pd.read_csv('/home/technonia/intern/affectnet_2/validation_0112.csv') # 파일명을 본인의 파일명으로 수정하세요

# df['Label_x1'] = df['face_x']
# df['Label_y1'] = df['face_y']
# df['Label_x2'] = df['face_x'] + df['face_width']
# df['Label_y2'] = df['face_y'] + df['face_height']

# # print(df[['face_x', 'face_y', 'face_width', 'face_height', 'Label_x1', 'Label_y1', 'Label_x2', 'Label_y2']].head())

# df.to_csv('update2_validation_0112.csv', index=False)
# ---------------------------------------------

from torchvision.transforms import Compose, Grayscale, Resize
from feat.utils.image_operations import BBox
from src.faceinsight.engine.chain.val_aro_image_chain import ValenceArousalImageChain
from src.faceinsight.engine.postprocessor.emotion_calibration_utils import get_dominant_emotion

def custom_batch_make(self, frame, detected_face, *args, **kwargs):
    """
    모든 detected face에 대해 사진을 저장하도록 수정됨.
    if i == 0 조건을 제거하여 모든 i(얼굴 인덱스)에 대해 저장 로직이 돕니다.
    """

    SAVE_DEBUG_DIR = "/home/technonia/intern/faceinsight/close_img_faces" 
    if not os.path.exists(SAVE_DEBUG_DIR):
        os.makedirs(SAVE_DEBUG_DIR)

    raw_fname = getattr(self, "current_filename", "unknown")
    current_fname = os.path.basename(raw_fname) # 경로 제거
    if "." in current_fname: 
        current_fname = os.path.splitext(current_fname)[0] # 확장자 제거

    transform = Compose([Grayscale(3)])
    gray = transform(frame)

    if not detected_face:
        return None

    len_index = [len(aa) for aa in detected_face]
    length_cumu = np.cumsum(len_index)
    flat_faces = [item for sublist in detected_face for item in sublist]

    concat_batch = None
    
    for i, face in enumerate(flat_faces):
        frame_choice = np.where(i < length_cumu)[0][0]
        bbox = BBox(face[:-1])
        
        face_tensor = (
            bbox.expand_by_factor(1.1)
            .extract_from_image(gray[frame_choice])
            .unsqueeze(0)
        )

        try:
            face_cpu = face_tensor.clone().cpu()
            
            # 차원 정리 (B, C, W, H) -> (C, W, H)
            if face_cpu.dim() == 4: 
                face_cpu = face_cpu.squeeze(0)
            
            # (C, W, H) -> (W, H, C)
            if face_cpu.dim() == 3:
                face_permuted = face_cpu.permute(1, 2, 0)
                face_np = face_permuted.numpy()
            elif face_cpu.dim() == 2:
                face_np = face_cpu.numpy()
            else:
                face_np = None

            if face_np is not None and face_np.size > 0:
                if face_np.max() <= 1.0:
                    face_np = (face_np * 255).astype(np.uint8)
                else:
                    face_np = face_np.astype(np.uint8)

                save_name = f"{current_fname}_face{i}.png"
                save_path = os.path.join(SAVE_DEBUG_DIR, save_name)
                
                if face_np.ndim == 3 and face_np.shape[2] == 3:
                    cv2.imwrite(save_path, face_np[:, :, ::-1]) 
                else:
                    cv2.imwrite(save_path, face_np)
                    
        except Exception as e:
            print(f"저장 실패: {e}") # 필요시 주석 해제
            pass

        # 텐서 병합 (모델 입력용)
        transform_resize = Resize(self.image_size)
        face_tensor = transform_resize(face_tensor) / 255
        if concat_batch is None:
            concat_batch = face_tensor
        else:
            concat_batch = torch.cat((concat_batch, face_tensor), 0)

    return concat_batch

def run_csv_analysis(csv_path, image_base_path, chain, emotion_classes):
    # 불러오는 애들
    cols_to_use = ['filename', 'Label_x1', 'Label_y1', 'Label_x2', 'Label_y2', 'expression']
    df = pd.read_csv(csv_path, usecols=cols_to_use)

    # input_label_map = { #affectnet csv 라벨 수정
    #     0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 
    #     4: 'fear', 5: 'disgust', 6: 'angry'
    # }
    # df['expression'] = df['expression'].map(input_label_map)

    output_label_map = { #출력되는 거 수정
        0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 
        4: 'sad', 5: 'surprise', 6: 'neutral'
    }
    
    results_data = []
    print(f"총 {len(df)}개의 데이터를 분석합니다.")


    model_instance = chain.au_agent.detector.emotion_model
    model_instance._batch_make = types.MethodType(custom_batch_make, model_instance)
    print("이미지 저장 가능~")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing CSV"):
        filename = row['filename']
        img_full_path = os.path.join(image_base_path, filename)
        
        chain.au_agent.detector.emotion_model.current_filename = filename #여기 batch mask

        input_image = cv2.imread(img_full_path)

        file_exists = os.path.exists(img_full_path)
        
        analysis_res = {
            "empty": 0 if file_exists else 1, # 파일 찾으면 0, 못찾으면 1
            "is_detected": False,
            "count_faces": 0,
            "img_size": None, # input 사진 해상도
            # "esti_x1": np.nan, "esti_y1": np.nan, 
            # "esti_x2": np.nan, "esti_y2": np.nan,
            # "esti_confidence": np.nan,
            "esti_expression": "", # 최종 결정 감정 
            "esti_score": None,  # 최종 결정된 감정의 점수

            # 각 감정별 점수
            "angry": None, "disgust": None, "fear": None, 
            "happy": None, "sad": None, "surprise": None, "neutral": None,

            "diff_center_x": None,
            "diff_center_y": None,
            # bounding box와 label x,y 값 중심점 비교

            # "box_mean": None, #중심점간의 평균
            # "box_var": None, #중심점간의 분산
            "area_result": None,
            "label_area": None,
            "estimate_area": None
        }

        if input_image is not None:
            h, w = input_image.shape[:2]
            analysis_res["img_size"] = f"{w}*{h}" #해상도 저장

            image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # 여기서 사진
            _, _, faces, _, emotions, _ = chain.au_agent.run(image_rgb, ndarray_to_list=False) #에러나서 추가

            # faces가 [[]] 처럼 비어있는 경우를 방지하기 위해 2단계 체크 아오오 에러나서
            if faces is not None and len(faces) > 0 and len(faces[0]) > 0:
                if isinstance(faces[0][0], (list, np.ndarray)):
                    actual_faces = faces[0]
                else:
                    actual_faces = faces
               
                # 실제 얼굴 데이터 있없 확인
                if len(actual_faces) > 0 and len(actual_faces[0]) >= 5:
                    analysis_res["is_detected"] = True
                    analysis_res["count_faces"] = len(actual_faces)
                    
                    ex1, ey1, ex2, ey2, econf = actual_faces[0]
                    analysis_res["e_x1"], analysis_res["e_y1"] = int(ex1), int(ey1)
                    analysis_res["e_x2"], analysis_res["e_y2"] = int(ex2), int(ey2)
                    analysis_res["e_confidence"] = round(float(econf), 3)

                    # ex1, ey1, ex2, ey2, e_confidence = actual_faces[0]
                    # analysis_res.update({
                    #     "e_x1": int(ex1), "e_y1": int(ey1), 
                    #     "e_x2": int(ex2), "e_y2": int(ey2), 
                    #     "e_confidence": round(float(e_confidence), 3)
                    # })

                    # analysis_res[f"face"] = actual_faces[0] #이게 confidence 제일 높으니까 이것만 일단 사용
                    if len(actual_faces) > 0:
                        for i in range(len(actual_faces)): #얼굴 개수만큼 반복 (actual_faces)
                            analysis_res[f"face_{i}"] = actual_faces[i]
                    

                    # 중심점끼리 var, mean 비교
                    l_cx = (row['Label_x1'] + row['Label_x2']) / 2
                    l_cy = (row['Label_y1'] + row['Label_y2']) / 2
                    e_cx = (ex1 + ex2) / 2
                    e_cy = (ey1 + ey2) / 2
                    
                    # diffs = np.array([abs(l_cx - e_cx), abs(l_cy - e_cy)])
                    d_cx= abs(l_cx - e_cx) #라벨 중심 - 결과 중심 x
                    d_cy = abs(l_cy - e_cy)
                    analysis_res["diff_center_x"] = d_cx
                    analysis_res["diff_center_y"] = d_cy

                    l_area = (row['Label_x2']-row["Label_x1"]) * (row["Label_y2"]-row["Label_y1"])
                    e_area = (ex2-ex1)*(ey2-ey1)
                    analysis_res["label_area"] = l_area
                    analysis_res["estimate_area"] = e_area
                    analysis_res["area_result"] = l_area - e_area
                    # analysis_res["box_mean"] = np.mean(diffs)
                    # analysis_res["box_var"] = np.var(diffs)


                    # 감정 분석 시작~~~~
                    if emotions is not None and len(emotions) > 0:
                        probs = emotions
                        while isinstance(probs, (list, np.ndarray)) and len(probs) > 0 and isinstance(probs[0], (list, np.ndarray)):
                            probs = probs[0]
                        
                        if isinstance(probs, (list, np.ndarray)) and len(probs) > 0:
                            dom_emotion_name, dom_score = get_dominant_emotion(probs, emotion_classes)
                            if dom_emotion_name in emotion_classes:
                                expr_idx = emotion_classes.index(dom_emotion_name)
                                analysis_res["esti_expression"] = output_label_map.get(expr_idx, "unknown")
                    
                                analysis_res["esti_score"] = round(dom_score, 3)
                                analysis_res["angry"] = round(probs[0], 3)
                                analysis_res["disgust"] = round(probs[1], 3)
                                analysis_res["fear"] = round(probs[2], 3)
                                analysis_res["happy"] = round(probs[3], 3)
                                analysis_res["sad"] = round(probs[4], 3)
                                analysis_res["surprise"] = round(probs[5], 3)
                                analysis_res["neutral"] = round(probs[6], 3)

        combined_row = row.to_dict()
        combined_row.update(analysis_res)
        results_data.append(combined_row)

    return pd.DataFrame(results_data)

if __name__ == "__main__":
    IMAGE_BASE_PATH = "/home/technonia/intern/faceinsight/close_img_0115" #이 폴더가 True + empty0인 사진만 있는 폴더
    INPUT_CSV_PATH = "/home/technonia/intern/faceinsight/validation_close_img_0115.csv" #validation_0113.csv가 detect True + empty 0만 담아둔 csv
    OUTPUT_CSV_PATH = "0115_affectnet_result_big_img_test.csv"

    chain = ValenceArousalImageChain()
    emotion_classes = chain.au_agent.emotion_labels

    if os.path.exists(INPUT_CSV_PATH):
        df_final = run_csv_analysis(INPUT_CSV_PATH, IMAGE_BASE_PATH, chain, emotion_classes)
        df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig', float_format='%.3f')
        
        print("\n" + "="*50)
        
        # 얼굴이 검출된 데이터만 필터링하여 계산 (None 제외)
        detected_df = df_final[df_final["is_detected"] == True]

        all_diff_values = pd.concat([detected_df["diff_center_x"], detected_df["diff_center_y"]])

        if not detected_df.empty:
            # d_cx d_cy
            diff_cx = detected_df["diff_center_x"]
            diff_cy = detected_df["diff_center_y"]
            print(f"\ndiff_center_x의 mean: {diff_cx.mean():.3f}")
            print(f"diff_center_x의 var: {diff_cx.var():.3f}")
            print(f"diff_center_x의 중앙값: {diff_cx.median():.3f}")

            print(f"\ndiff_center_y의 mean: {diff_cy.mean():.3f}")
            print(f"diff_center_y의 var: {diff_cy.var():.3f}")
            print(f"diff_center_y의 중앙값: {diff_cy.median():.3f}")

            """
            analysis_res["label_area"] = l_area
                    analysis_res["estimate_area"] = e_area
                    analysis_res["area_result"] = l_area - e_area
            """
            area_result = detected_df["area_result"]
            print(f"\n(label - estimate area)의 mean: {area_result.mean():.3f}")
            print(f"(label - estimate area)의 var: {area_result.var():.3f}")

            # area_result 할 때 양수 음수 나눠서 mean, var 계산하기 
            neg_group = area_result[area_result < 0]
            zero_group = area_result[area_result == 0]
            pos_group = area_result[area_result > 0]
            if not neg_group.empty:
                print(f"\n(label - estimate area) 음수인 경우 mean: {neg_group.mean():.3f}")
                print(f"(label - estimate area)음수인 경우 var: {neg_group.var():.3f}")
            if not pos_group.empty:
                print(f"\n(label - estimate area)양수인 경우 mean: {pos_group.mean():.3f}")
                print(f"(label - estimate area)양수인 경우 var: {pos_group.var():.3f}")
            
            if not zero_group.empty:
                print(f"\n(label - estimate area) == 0인 경우 개수 : {len(zero_group)}개")
            else:
                print("\n(label - estimate area) == 0인 경우 없음!")
            

            # print("감정별 area result: ")            
            # emotion_area_mean = detected_df.groupby("expression")["area_result"].mean().sort_values()
            # print(emotion_area_mean)
            
        else:
            print("분석 가능한 얼굴 검출 데이터가 없습니다.")

    
        # if not detected_df.empty:
        #     print("\n" + "-"*50)
        #     print("⚠️ 오차가 가장 큰(Top 10) 데이터 리스트 (이상치 의심)")
        #     print("-"*50)
            
        #     # box_mean 기준 내림차순 정렬
        #     top_outliers = detected_df.sort_values(by="box_mean", ascending=False).head(10)
            
        #     for i, row in top_outliers.iterrows():
        #         print(f"파일: {row['subDirectory_filePath']}")
        #         print(f"   ㄴ 오차(mean): {row['box_mean']:.2f} / label 감정: {row['expression']} / esti감정: {row['esti_expression']}")
        #     print("-"*50)
            
        print("="*50)
        print(f"분석 완료 및 저장됨: {OUTPUT_CSV_PATH}")
    else:
        print(f"에러: {INPUT_CSV_PATH} 파일을 찾을 수 없습니다.")