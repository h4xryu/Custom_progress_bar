import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

class ECGClassifier:
    """ECG 분류 모델 클래스"""
    def __init__(self, model_path, model_type='onnx'):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.session = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
        
    def _load_model(self):
        """모델 로드"""
        try:
            if self.model_type == 'onnx':
                self.session = ort.InferenceSession(str(self.model_path))
            elif self.model_type == 'pytorch':
                self.model = torch.load(self.model_path, map_location=self.device)
                self.model.eval()
            else:
                raise ValueError(f"지원하지 않는 모델 형식입니다: {self.model_type}")
                
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")
            
    def predict(self, segments):
        """세그먼트 분류"""
        try:
            if self.model_type == 'onnx':
                return self._predict_onnx(segments)
            else:
                return self._predict_pytorch(segments)
                
        except Exception as e:
            raise Exception(f"예측 실패: {str(e)}")
            
    def _predict_onnx(self, segments):
        """ONNX 모델로 예측"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: segments.astype(np.float32)})
        return outputs[0]
        
    def _predict_pytorch(self, segments):
        """PyTorch 모델로 예측"""
        with torch.no_grad():
            segments = torch.FloatTensor(segments).to(self.device)
            outputs = self.model(segments)
            return outputs.cpu().numpy()
            
    def get_class_names(self):
        """클래스 이름 반환"""
        return [
            "정상",
            "심방 조기 수축",
            "심실 조기 수축",
            "심방 세동",
            "심실 세동",
            "심실 빈맥",
            "심실 서맥",
            "심실 빈맥",
            "심실 세동",
            "심실 빈맥",
            "심실 세동",
            "심실 빈맥",
            "심실 세동",
            "심실 빈맥",
            "심실 세동"
        ] 