import os
import cv2
import numpy as np
from tqdm import tqdm

"""
batch mask에서 추출한 face[0] 일 때 이미지랑 원본 이미지랑 병합된 이미지를 만들고자 .,,
"""
def create_comparison_view(original_dir, debug_dir, save_dir):
    """
    """
    
    # 결과 저장 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 저장 폴더 생성됨: {save_dir}")

    # debug 폴더에 있는 파일 리스트 가져오기
    if not os.path.exists(debug_dir):
        print(f"❌ 에러: debug 폴더가 없습니다 -> {debug_dir}")
        return

    debug_files = [f for f in os.listdir(debug_dir) if f.endswith(".png")]
    
    print(f"🔍 총 {len(debug_files)}개의 Crop 이미지를 찾았습니다. 병합을 시작합니다...")

    success_count = 0

    for debug_filename in tqdm(debug_files, desc="Merging Images"):
        # -------------------------------------------------------------
        # 1. 파일명 파싱 (원본 파일명 추출)
        # -------------------------------------------------------------
        suffix = "_debug_resmasknet_face_0.png" # 우리가 뒤에 붙인 이름
        
        # 파일명이 우리가 정한 형식인지 확인
        if suffix not in debug_filename:
            continue
            
        # 접미사를 제거하여 원본 파일명 획득
        original_filename = debug_filename.replace(suffix, "")
        
        # -------------------------------------------------------------
        # 2. 이미지 불러오기 (끊겼던 부분 시작)
        # -------------------------------------------------------------
        path_org = os.path.join(original_dir, original_filename)
        path_crop = os.path.join(debug_dir, debug_filename)

        # 원본 파일이 있는지 확인
        if not os.path.exists(path_org):
            continue

        img_org = cv2.imread(path_org)
        img_crop = cv2.imread(path_crop)

        if img_org is None or img_crop is None:
            continue

        # -------------------------------------------------------------
        # 3. 이미지 크기 맞추기 & 붙이기
        # -------------------------------------------------------------
        h_org, w_org = img_org.shape[:2]
        h_crop, w_crop = img_crop.shape[:2]
        
        if h_crop == 0: continue

        # 원본 높이에 맞춰서 Crop 이미지를 확대 (비율 유지)
        scale = h_org / h_crop
        new_w_crop = int(w_crop * scale)
        img_crop_resized = cv2.resize(img_crop, (new_w_crop, h_org))

        # [원본] - [구분선] - [확대된 Crop]
        combined_img = np.hstack((img_org, img_crop_resized))

        # -------------------------------------------------------------
        # 4. 저장하기
        # -------------------------------------------------------------
        save_name = f"{original_filename}"
        # 원본 이름에 확장자가 없는 경우를 대비해 jpg 붙이기 (보통은 있음)
        if not save_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            save_name += ".jpg"
            
        cv2.imwrite(os.path.join(save_dir, save_name), combined_img)
        success_count += 1

    print("="*50)
    print(f"✅ 작업 완료! 총 {success_count}장의 비교 이미지가 생성되었습니다.")
    print(f"📂 저장 경로: {save_dir}")

if __name__ == "__main__":
    ORIGINAL_DIR = "/home/technonia/intern/faceinsight/validation_csv_img"
    DEBUG_DIR = "/home/technonia/intern/faceinsight/debug_image" #batch mask 돌렸을때
    RESULT_DIR = "/home/technonia/intern/faceinsight/debug&org_img" #저장할곳

    # 함수 실행 (RESULT_DIR가 save_dir로 전달됨)
    create_comparison_view(ORIGINAL_DIR, DEBUG_DIR, RESULT_DIR)
