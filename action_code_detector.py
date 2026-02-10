from feat import Detector
from feat.plotting import plot_face
# from feat.utils import set_torch_device
from feat.utils.io import get_resource_path
from feat.utils.image_operations import convert_image_to_tensor

import numpy as np
import os
import torch
from time import time
import multiprocessing
from scipy.spatial import distance
from imutils import face_utils
import xgboost as xgb

# custom modules
from faceinsight.reader.base import BaseReader
from faceinsight.engine.facial_landmark.base import BaseLandmarkAdapter
from faceinsight.db.base import BaseDatabase, DummyDatabase
from faceinsight.utils.logger import Logger
from faceinsight.constants.agent.pyfeat_au import PYFEAT_AU_LIST

from .base import BaseAgent


# Finding landmark id for left and right eyes
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.28 # 이 값 이상이면 눈 뜬걸로 (원본 0.28)
logger = Logger()


class PyFeatActionCodeDetectionAgent(BaseAgent):
    def __init__(
        self,
        reader: BaseReader,
        landmark_adapter: BaseLandmarkAdapter = BaseLandmarkAdapter(),
        db: BaseDatabase = DummyDatabase(),
        need_ear: bool = True,
        device: str = "auto"  # auto, cpu, cuda
    ) -> None:
        self.reader = reader
        self.landmark_adapter = landmark_adapter
        self.db = db
        self.need_ear = need_ear

        self.use_emotion_detection = True  # whether to use emotion detection or not
        self.use_au_detection = False  # whether to use action unit detection or not #TODO: au랑 landmark 안 쓰니까 false로 바꾸기

        # Set the device to GPU if available
        self.cuda_available = torch.cuda.is_available()
        device = "cuda" if self.cuda_available else "cpu"
        self.device = device

        # num of cpus
        n_cpus = multiprocessing.cpu_count()
        if n_cpus > 8:
            n_jobs = 8
        else:
            n_jobs = n_cpus

        self.n_cpus = n_cpus
        self.n_jobs = n_jobs

        # Define the function just to extract landmarks from images
        # <https://github.com/cosanlab/py-feat/blob/v0.6.2/docs/pages/models.md>
        self.detector = Detector(
            face_model='retinaface',
            emotion_model='resmasknet',
            landmark_model="mobilefacenet",
            facepose_model='img2pose',
            au_model='xgb', # xgb, svm
            device=self.device,
            n_jobs=n_jobs,  # use as much jobs as possible to speed up the detection
        )

        self.emotion_labels = ['anger','disgust','fear','happiness','sadness','surprise','neutral']
        self.emotion_model = self.detector.emotion_model

        self.au_keys = PYFEAT_AU_LIST
        self.au_classifiers = {}
        for keys in self.au_keys:
            classifier = xgb.XGBClassifier()
            classifier.load_model(
                os.path.join(get_resource_path(), f"July4_{keys}_XGB.ubj")
            )

            # if gpu is available, load the model to gpu
            if self.cuda_available:
                classifier.tree_method = 'gpu_hist'

            self.au_classifiers[keys] = classifier


    @torch.inference_mode()
    def run(self, frame, detect_pose: bool = False, ndarray_to_list: bool = True, face_threshold: float = 0.25):
        frame_tch = torch.from_numpy(frame).to(self.device)
        cpu_frame_tch = torch.from_numpy(frame).cpu()
        # convert from [w, h, c] to [c, h, w]
        frame_tch = frame_tch.permute(2, 0, 1).unsqueeze(0)  # add batch dimension
        cpu_frame_tch = cpu_frame_tch.permute(2, 0, 1).unsqueeze(0)  # add batch dimension

        total_start = time()

        faces = self.detector.detect_faces(frame, threshold=0.5)
        face_detect_time = time() - total_start

        # landmark 감지
        landmark_start = time()
        landmarks = self.detector.detect_landmarks(frame_tch, detected_faces=faces)
        landmark_time = time() - landmark_start

        # EAR 계산
        ear_calc_start = time()
        if landmarks is None or len(landmarks) == 0 or len(landmarks[0]) == 0:
            ear = {'eye': 0.0, 'left_eye': 0.0, 'right_eye': 0.0}
        else:
            ear = self.get_ear(landmarks[0][0])
        ear_calc_time = time() - ear_calc_start

        ear['landmark_time'] = landmark_time
        ear['ear_calc_time'] = ear_calc_time
        ear['landmark_ear_time'] = landmark_time + ear_calc_time
        ear['face_detect_time'] = face_detect_time

        # else:
        #     landmarks = None
        #     ear = -1
        # t3 = time()
        # 원본
        # if self.use_au_detection:
        #     action_units = self.detector.detect_aus(cpu_frame_tch, landmarks=landmarks)
        # else:
        #     action_units = []

        # t4 = time()
        
        # 0116 수정~ 알아서 해상도만큼 사이즈 가져오도록!--------
        h, w = frame.shape[:2] 
        #faces = [[[0, 0, w, h, 1.0]]]
        # -----------------------------------------------------

        # faces = self.detector.detect_faces(frame, threshold=0.5) #원본 

        emotion_start = time()
        if self.use_emotion_detection:
            emotions = self.detector.detect_emotions(frame_tch, faces, [])
        else:
            emotions = []
        emotion_time = time() - emotion_start

        ear['emotion_time'] = emotion_time
        ear['total_time'] = time() - total_start

        # 0112 수정 ---------------------------------------------------
        # 이게 원본
        # if ndarray_to_list:
        #     if self.use_emotion_detection:
        #         emotions = [emotion.tolist() for emotion in emotions]

        # 
        # if ndarray_to_list:
        #     if self.use_emotion_detection:
        #         # 데이터가 넘파이 배열일 때만 tolist()를 호출하도록 변경
        #         emotions = [e.tolist() if hasattr(e, 'tolist') else e for e in emotions]
        # ---------------------------------------------------

        # 0121 수정 - 얼굴 감지 안 되는 애들 처리
        if ndarray_to_list:
            if self.use_emotion_detection and emotions:
                emotions = [
                    emotion.tolist() if hasattr(emotion, 'tolist') else emotion 
                    for emotion in emotions
                ]
        # -------------------------------------------

        return [], landmarks, faces, [], emotions, ear


    @torch.inference_mode()
    def run_with_optimized(self, frame, detect_pose: bool = False, ndarray_to_list: bool = True):
        frame_tch = torch.from_numpy(frame).to(self.device)
        cpu_frame_tch = torch.from_numpy(frame).cpu()
        # convert from [w, h, c] to [c, h, w]
        frame_tch = frame_tch.permute(2, 0, 1).unsqueeze(0)  # add batch dimension
        cpu_frame_tch = cpu_frame_tch.permute(2, 0, 1).unsqueeze(0)  # add batch dimension

        faces = self.detector.detect_faces(frame, threshold=0.5)

        # for robustness, if no face is detected, return empty lists
        if faces is None or len(faces) == 0:
            return [], [], [], None, [], {}
        if len(faces[0]) == 0:
            return [], [], [], None, [], {}

        t1 = time()
        if self.use_au_detection:
            landmarks = self.detector.detect_landmarks(frame_tch, detected_faces=faces)
            ear = self.get_ear(landmarks[0][0])
        else:
            landmarks = None
            ear = -1

        t2 = time()
        if detect_pose:
            poses = self.detector.detect_facepose(cpu_frame_tch, landmarks=landmarks)
        else:
            poses = None
        t3 = time()
        if self.use_au_detection:
            # action_units = self.detector.detect_aus(cpu_frame_tch, landmarks=landmarks)
            cpu_frame_tch_ = convert_image_to_tensor(cpu_frame_tch, img_type="float32")
            cpu_frame_tch_, landmarks_ = self.detector._batch_hog(
                frames=cpu_frame_tch_, landmarks=landmarks
            )

            landmarks_ = np.concatenate(landmarks_)
            landmarks_ = landmarks_.reshape(-1, landmarks_.shape[1] * landmarks_.shape[2])

            pca_transformed_upper = self.detector.au_model.pca_model_upper.transform(
                self.detector.au_model.scaler_upper.transform(cpu_frame_tch_)
            )
            pca_transformed_lower = self.detector.au_model.pca_model_lower.transform(
                self.detector.au_model.scaler_lower.transform(cpu_frame_tch_)
            )
            pca_transformed_full = self.detector.au_model.pca_model_full.transform(
                self.detector.au_model.scaler_full.transform(cpu_frame_tch_)
            )

            pca_transformed_upper = np.concatenate((pca_transformed_upper, landmarks_), 1)
            pca_transformed_lower = np.concatenate((pca_transformed_lower, landmarks_), 1)
            pca_transformed_full = np.concatenate((pca_transformed_full, landmarks_), 1)

            pred_aus = []
            for keys in self.au_keys:
                classifier = self.au_classifiers[keys]

                if keys in {"AU1", "AU2", "AU7"}:
                    au_pred = classifier.predict_proba(pca_transformed_upper)[:, 1]
                elif keys in {"AU11", "AU14", "AU17", "AU23", "AU24", "AU26"}:
                    au_pred = classifier.predict_proba(pca_transformed_lower)[:, 1]
                elif keys in {
                    "AU4", "AU5", "AU6", "AU9", "AU10",
                    "AU12", "AU15", "AU20", "AU25", "AU28", "AU43",
                }:
                    au_pred = classifier.predict_proba(pca_transformed_full)[:, 1]
                else:
                    raise ValueError("unknown AU detected")

                pred_aus.append(au_pred)
            action_units = np.array(pred_aus).T
        else:
            action_units = []

        t4 = time()
        if self.use_emotion_detection:
            batch = self.emotion_model._batch_make(frame=frame_tch, detected_face=faces)
            output = self.emotion_model.model(batch)
            proba = torch.softmax(output, 1)
            emotions = proba.cpu().numpy()
        else:
            emotions = []
        t5 = time()
        logger.log_info(
            f"Detection times: "
            f"Landmarks: {t2 - t1:.4f}s, "
            f"Pose: {t3 - t2:.4f}s, "
            f"AUs: {t4 - t3:.4f}s, "
            f"Emotions: {t5 - t4:.4f}s"
        )

        if ndarray_to_list:
            if self.use_au_detection:
                action_units = [au.tolist() for au in action_units]
            if self.use_emotion_detection:
                emotions = [emotion.tolist() for emotion in emotions]

        return action_units, landmarks, faces, poses, emotions, ear


    @torch.inference_mode()
    def run_with_debug(self, frame, detect_pose: bool = False, ndarray_to_list: bool = False):
        start_time = time()
        faces = self.detector.detect_faces(frame, threshold=0.5)
        face_elapsed = time() - start_time
        if faces is None or len(faces) == 0:
            return [], [], [], None, []
        if len(faces[0]) == 0:
            return [], [], [], None, []

        start_time = time()
        landmarks = self.detector.detect_landmarks(frame, detected_faces=faces)
        landmark_elapsed = time() - start_time
        ear = self.get_ear(landmarks[0][0])

        if detect_pose:
            poses = self.detector.detect_facepose(frame, landmarks=landmarks)
        else:
            poses = None

        start_time = time()
        action_units = self.detector.detect_aus(frame, landmarks=landmarks)
        action_unit_elapsed = time() - start_time

        start_time = time()
        emotions = self.detector.detect_emotions(frame, faces, landmarks)
        emotion_elapsed = time() - start_time

        if ndarray_to_list:
            print(action_units)
            action_units = [au.tolist() for au in action_units]
            emotions = [emotion.tolist() for emotion in emotions]

        print(" ** Detection times **")
        print("  > Face detection:", face_elapsed)
        print("  > Landmark detection:", landmark_elapsed)
        print("  > EAR detection:", ear)
        print("  > Action unit detection:", action_unit_elapsed)
        print("  > Emotion detection:", emotion_elapsed)
        print("  > Total:", time() - start_time)

        action_unit = action_units[0][0]
        # print(action_unit)
        ax = plot_face(au=action_unit, muscles=True)

        # pyplot to image
        # ax.figure.canvas.draw()
        cnt = self.reader.get_count()
        ax.figure.savefig(f"debug/{cnt}.png")

        return action_units, landmarks, faces, poses, emotions, ear


    def calculate_arousal(self, action_units_for_arousal: list[float]):
        action_units_for_arousal_ = action_units_for_arousal[:-1]

        p_cnt = len(action_units_for_arousal_)
        action_unit_val_p = sum(action_units_for_arousal_)
        action_unit_val_n = action_units_for_arousal[-1] * p_cnt

        # assume that the last action unit AU43 is already inverted
        action_unit_val = action_unit_val_p + action_unit_val_n
        action_unit_val /= p_cnt

        return action_unit_val


    def calculate_valence(self, emotions) -> float:
        #valence : 감정의 부정 || 긍정적인 정도
        happy = emotions[3]

        anger = emotions[0]
        disgust = emotions[1]
        fear = emotions[2]
        sadness = emotions[4]

        # surprise = emotions[5]
        # neutral = emotions[6]

        max_negative = max(anger, disgust, fear, sadness)
        max_positive = happy

        valence = max_positive - max_negative
        return valence


    def get_ear(self, landmarks):
        leftEye = landmarks[leftEyeStart:leftEyeEnd]
        rightEye = landmarks[rightEyeStart:rightEyeEnd]

        leftEye = eye_aspect_ratio(leftEye)
        rightEye = eye_aspect_ratio(rightEye)

        eye = (leftEye + rightEye) / 2.0
        ear = {
            'eye': eye,
            'left_eye': leftEye,
            'right_eye': rightEye
        }

        return ear


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    C = distance.euclidean(eye[0], eye[3])
    eye = (A + B) / (2.0 * C)

    return eye
