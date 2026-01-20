# colab으로 affectnet 학습 시킨 거 inference.py
# affectnet 논문 상의 감정 순서와 동일하도록~
# 0 Neutral, 1 Happy, 2 Sad, 3 Surprise, 4 Fear, 5 Disgust, 6 Anger
# colab 코드 실행했을 때 output은 .pt 파일?!

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "/home/technonia/intern/faceinsight/models/cage/model.pt" #경로 수정완 
INPUT_CSV_PATH = "/home/technonia/intern/faceinsight/no_close/validation_no_close_img_0116.csv" #경로 수정
IMAGE_COLUMN_NAME = "filename" 
BASE_IMAGE_FOLDER = "/home/technonia/intern/faceinsight/no_close/no_close_img_0116" #이미지 경로
OUTPUT_CSV_PATH = "0120_colab_result.csv" #저장할 파일명


EMOTIONS = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger'}

def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = timm.create_model("maxvit_tiny_tf_224", pretrained=False, num_classes=9)
    except Exception as e:
        print(f"모델 생성 에러: {e}")
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('blocks', 'stages')
        new_state_dict[new_k] = v

    model.load_state_dict(state_dict, strict=False)

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"가중치 로드 결과: {msg}") 

    model.to(device)
    model.eval()
    return model


def process_image(image_path):
    # resize
    # transforms.ToTensor(),  # saves image as tensor (automatically divides by 255)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return None # 이미지 못 찾으면 None


def run_batch_inference():
    print(f"데이터 로드ㅡ {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)

        label_mapping = {'anger': 'angry', 'happiness': 'happy'} #여기서 가져오는 거 변경
        if 'expression' in df.columns:
            df['expression'] = df['expression'].replace(label_mapping)

    except Exception as e:
        print(f"CSV를 찾을 수 없습니다: {e}")
        return

    model = load_trained_model(MODEL_PATH)
    if model is None: return

    print(f"총 {len(df)}개의 분석을 시작합니다...")

    # [수정] 딕셔너리 키 이름을 'esti_'로 통일하여 혼동 방지
    results = {
        'esti_expression': [], # 예측된 감정
        'esti_score': [],

        'esti_valence': [],
        'esti_arousal': [],

        'esti_neutral': [],
        'esti_happy': [],
        'esti_sad': [],
        'esti_surprise': [],
        'esti_fear': [],
        'esti_disgust': [],
        'esti_anger': []
    }
    
    emotion_keys = ['esti_neutral', 'esti_happy', 'esti_sad', 'esti_surprise', 'esti_fear', 'esti_disgust', 'esti_anger']

    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        img_filename = row[IMAGE_COLUMN_NAME]
        full_path = os.path.join(BASE_IMAGE_FOLDER, img_filename)

        input_tensor = process_image(full_path)

        if input_tensor is None: 
            results['esti_expression'].append("Error")
            results['esti_score'].append(0.0)
            results['esti_valence'].append(0.0)
            results['esti_arousal'].append(0.0)
            
            for key in emotion_keys:
                results[key].append(0.0)
            continue

        with torch.no_grad():
            outputs = model(input_tensor)
            
            if outputs.shape[1] == 9:
                outputs_cls = outputs[:, :7]
                outputs_reg = outputs[:, 7:]
            else:
                outputs_cls = outputs
                outputs_reg = torch.zeros(1, 2).to(DEVICE)

            probs = torch.nn.functional.softmax(outputs_cls, dim=1)
            top_prob, top_class_idx = torch.max(probs, 1)
            
            results['esti_expression'].append(EMOTIONS[top_class_idx.item()])
            results['esti_score'].append(round(top_prob.item(), 3))
            
            results['esti_valence'].append(round(outputs_reg[0][0].item(), 3))
            results['esti_arousal'].append(round(outputs_reg[0][1].item(), 3))

            results['esti_neutral'].append(round(probs[0][0].item(), 3))
            results['esti_happy'].append(round(probs[0][1].item(), 3))
            results['esti_sad'].append(round(probs[0][2].item(), 3))
            results['esti_surprise'].append(round(probs[0][3].item(), 3))
            results['esti_fear'].append(round(probs[0][4].item(), 3))
            results['esti_disgust'].append(round(probs[0][5].item(), 3))
            results['esti_anger'].append(round(probs[0][6].item(), 3))

    df['esti_expression'] = results['esti_expression']
    df['esti_score'] = results['esti_score']
    
    df['esti_valence'] = results['esti_valence']
    df['esti_arousal'] = results['esti_arousal']

    df['esti_neutral'] = results['esti_neutral']
    df['esti_happy'] = results['esti_happy'] 
    df['esti_sad'] = results['esti_sad']
    df['esti_surprise'] = results['esti_surprise']
    df['esti_fear'] = results['esti_fear']
    df['esti_disgust'] = results['esti_disgust']
    df['esti_anger'] = results['esti_anger']

    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n완료: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_batch_inference()