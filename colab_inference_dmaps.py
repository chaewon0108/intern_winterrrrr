# dmaps 영상에서 캡쳐된 사진 사용할 때
# dmaps 영상 캡쳐 사진은 아직 labeling 된 csv가 없어서 label과 estimate를 비교할 수 없음!

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = "/home/technonia/intern/faceinsight/models/cage/model.pt"
IMAGE_FOLDER_PATH = "/home/technonia/intern/dmaps_data_capture_img"
OUTPUT_CSV_PATH = "0120_colab_dmaps_result.csv"
BATCH_SIZE = 128

print("batch size : ", BATCH_SIZE)

EMOTIONS = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'angry'}

def load_trained_model(model_path):    
    if not os.path.exists(model_path):
        print(f"모델 파일 없음: {model_path}")
        return None
        
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    try:
        model = models.maxvit_t(weights="DEFAULT")
    except Exception as e:
        print(f"model load 할 때 에러: {e}")
        return None

    in_features = model.classifier[5].in_features
    model.classifier[5] = nn.Linear(in_features, 9, bias=False)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if 'head.fc.' in k: new_k = k.replace('head.fc.', 'classifier.5.')
        elif 'head.' in k: new_k = k.replace('head.', 'classifier.5.')
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(valid_extensions)]
        self.image_files.sort() # 순서 고정

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_name
        except Exception:
            return torch.zeros((3, 224, 224)), "error_file"

def run_inference():
    model = load_trained_model(MODEL_PATH)
    if model is None: return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(IMAGE_FOLDER_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"데이터 :: {len(dataset)}개")
    all_probs = []
    all_va = []
    all_filenames = []

    with torch.no_grad():
        for images, filenames in tqdm(dataloader):
            images = images.to(DEVICE)
            outputs = model(images)

            if outputs.shape[1] == 9:
                outputs_cls = outputs[:, :7]
                outputs_reg = outputs[:, 7:]
            else:
                outputs_cls = outputs
                outputs_reg = torch.zeros(outputs.size(0), 2).to(DEVICE)

            probs = torch.nn.functional.softmax(outputs_cls, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_va.append(outputs_reg.cpu().numpy())
            all_filenames.extend(filenames)

    final_probs = np.concatenate(all_probs, axis=0) 
    final_va = np.concatenate(all_va, axis=0)
    top_indices = np.argmax(final_probs, axis=1)
    

    res_df = pd.DataFrame({'filename': all_filenames})
    res_df['esti_expression'] = [EMOTIONS[idx] for idx in top_indices]
    res_df['esti_score'] = np.round(np.max(final_probs, axis=1), 3)

    res_df['esti_valence'] = np.round(final_va[:, 0], 3)
    res_df['esti_arousal'] = np.round(final_va[:, 1], 3)

    emotion_cols = ['esti_neutral', 'esti_happy', 'esti_sad', 'esti_surprise', 'esti_fear', 'esti_disgust', 'esti_angry']
    for i, col in enumerate(emotion_cols):
        res_df[col] = np.round(final_probs[:, i], 3)

    res_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n완료띠~~: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_inference()
