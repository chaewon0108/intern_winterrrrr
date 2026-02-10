from typing import Union
from copy import deepcopy
import os
import torch
import cv2

# custom modules
from faceinsight.db.base import DummyDatabase
from faceinsight.reader.file_reader import VideoFileReader
from faceinsight.engine.agent.action_code_detector import PyFeatActionCodeDetectionAgent
from faceinsight.utils.logger import Logger
from faceinsight.constants.agent.pyfeat_au import EMOTION_CATS
from faceinsight.engine.postprocessor.emotion_calibration_utils import calibrate_emotion_with_neutral_reference
from faceinsight.ai.cage import load_cage_model, get_cage_transformer

from . import BaseChain
from .val_aro_graph_chain import ValenceArousalResult


logger = Logger()


# custom type
ValenceArousalResultType = Union[ValenceArousalResult, None]


class ValenceArousalImageChain(BaseChain):
    def __init__(
        self,
        cage_model_name: str = "model.pt",
        cage_device: str = "cpu", # 여기서 cpu
        save_image: bool = False,
        save_image_path: str = "",
    ):
        super().__init__()

        self.reader = VideoFileReader(None)
        self.db = DummyDatabase()
        self.au_agent = PyFeatActionCodeDetectionAgent(self.db)
        self.add_agent(self.au_agent)

        self.emotion_categories = EMOTION_CATS
        self.base_emotions = None

        self.save_image = save_image
        self.save_image_path = save_image_path
        if save_image and not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path, exist_ok=True)

        guid = self.reader.guid
        if save_image:
            dir_path = os.path.join(self.save_image_path, f"{guid}")
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        self.frame = None

        self.use_cage = True
        self._init_cage(cage_model_name, cage_device)
        logger.log_info(f"Using CAGE model: {cage_model_name} on device: {cage_device}")
       


    def _init_cage(self, cage_model_name: str, cage_device: str):
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _model_dir = os.path.join(_current_dir, "../../../../models/cage")
        _model_path = os.path.join(_model_dir, cage_model_name)

        cage = load_cage_model("maxvit", _model_path, device=cage_device)
        self.cage = cage
        self.cage_transformer = get_cage_transformer()
        self.cage_device = torch.device(cage_device)


    def _run_chain(self, frame, save_first_frame: bool = False, postfix_index: int = 0):
        one_sec_results = []

        # Run the action unit detection agent
        (
            action_units, landmarks, faces, poses, emotions, ear
        ) = self.au_agent.run(frame, detect_pose=False, ndarray_to_list=True)

        # if no face detected, append None
        # if action_units is None or action_units == []:
        #     one_sec_results.append(None)
        #     return one_sec_results

        # action_units = action_units[0][0]
        # landmarks = landmarks[0][0]
    
        # 0121 수정 - 얼굴 감지 안 됐을 때 ----------------------------------
        if faces is None or len(faces) == 0 or (len(faces) > 0 and len(faces[0]) == 0):
            one_sec_results.append(None)
            return one_sec_results

        faces = faces[0] # 얼굴이 감지된 경우에만

        # emotions도
        if emotions and len(emotions) > 0 and len(emotions[0]) > 0:
            emotions = emotions[0][0] #pyfeat 결과 여기서
        else:
            emotions = [0.0] * 7 #7개 다 0으로 채우기
        # ---------------------------------------------------------

        # faces = faces[0] #pyfeat 결과 여기서 
        # emotions = emotions[0][0] #pyfeat 결과 여기서


        if self.base_emotions is None:
            self.base_emotions = deepcopy(emotions)
        # calibrated_emotions = calibrate_emotion_with_neutral_reference(
        #     self.base_emotions, emotions, self.emotion_categories
        # )
        # valence = self.au_agent.calculate_valence(calibrated_emotions)

        # arousal_action_units = action_units[:8] + action_units[9:-1] + [0 - action_units[-1]]
        # arousal = self.au_agent.calculate_arousal(arousal_action_units)

        timestamp = 0

        result = ValenceArousalResult(
            action_units=action_units,
            landmarks=landmarks,
            faces=faces,
            poses=poses,
            emotions=emotions,
            calibrated_emotions=[],
            ear=ear,
            action_units_for_arousal=[],

            valence=0,
            arousal=0,

            timestamp=timestamp,
        )

        # # if using CAGE, use the forward pass to the cage model # 이게 원본------------
        # if self.use_cage:
        #     with torch.no_grad():
        #         # crop frame to face
        #         x1, y1, x2, y2,_ = faces[0]

        #         x1 = int(x1)
        #         y1 = int(y1)
        #         x2 = int(x2)
        #         y2 = int(y2)
        #         copied_frame = deepcopy(frame)
        #         copied_frame = copied_frame[y1:y2, x1:x2]

        #         cage_input = self.cage_transformer(copied_frame)
        #         cage_input = cage_input.unsqueeze(0).to(self.cage_device)

        #         cage_output = self.cage(cage_input)
        #         cage_output = cage_output.squeeze(0).cpu().numpy()
        #         # cage_valence = cage_output[0]
        #         # cage_arousal = cage_output[1]

        #     # cast from torch tensor to float
        #     cage_valence_f = 0 #float(cage_valence)
        #     cage_arousal_f = 0 #float(cage_arousal)

        #     result.cage_valence = cage_valence_f
        #     result.cage_arousal = cage_arousal_f
        # --------------------------------------------------------------------------

        # 0121 수정 : model.pt의cage 결과 뽑아야 하는데, 얼굴이 인식 안 되면 오류나서
        if self.use_cage:
            with torch.no_grad():
                x1, y1, x2, y2, _ = faces[0]
                
                copied_frame = deepcopy(frame)[int(y1):int(y2), int(x1):int(x2)]
                
                if copied_frame.size > 0:
                    cage_input = self.cage_transformer(copied_frame)
                    cage_input = cage_input.unsqueeze(0).to(self.cage_device)

                    cage_output = self.cage(cage_input)
                    cage_output_tensor = cage_output.squeeze(0)
                    
                    # 여기서 emotion
                    cage_emotions_probs = torch.softmax(cage_output_tensor[:7], dim=0)
                    result.cage_emotions = cage_emotions_probs.cpu().numpy()
                    
                    # valence, arousal 구하기
                    cage_output_np = cage_output_tensor.cpu().numpy()
                    
                    if len(cage_output_np) >= 9:#valence, arousal
                        cage_valence = cage_output_np[7] # cage니까 7,8이 v/a
                        cage_arousal = cage_output_np[8]
                        
                        cage_valence_f = float(cage_valence)
                        cage_arousal_f = float(cage_arousal)
                        
                        result.cage_valence = cage_valence_f
                        result.cage_arousal = cage_arousal_f
                    else: #7,8에 값 없으면
                        result.cage_valence = 0.0
                        result.cage_arousal = 0.0
        # ------------------------------------------------------------------------------------
                    

        if save_first_frame and self.save_image:
            guid = self.reader.guid
            frame_path = os.path.join(self.save_image_path, f"{guid}/{postfix_index}.jpg")
            cv2.imwrite(frame_path, frame)
            copied_frame_path = os.path.join(self.save_image_path, f"{guid}/{postfix_index}_crop.jpg")
            # resize copied_frame to 255x255
            copied_frame = cv2.resize(copied_frame, (255, 255))
            cv2.imwrite(copied_frame_path, copied_frame)

        one_sec_results.append(result)
        return one_sec_results


    def run(self) -> list[ValenceArousalResultType]:
        if self.frame is None:
            logger.log_error("No frame set. Please set a frame before running the chain.")
            return []

        results: list[ValenceArousalResultType] = self._run_chain(
            self.frame,
            save_first_frame=False,
        )

        logger.log_info("Finished processing image.")
        self.results = results

        return results


    def set_frame(self, frame):
        self.frame = frame
