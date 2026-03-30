import torch
import os
from model import TicTacToeNN

def convert_to_onnx():
    model = TicTacToeNN()
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "tictactoe_model_3999.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    dummy_input = torch.zeros((1, 3, 3), dtype=torch.float32)
    onnx_file_path = os.path.join(current_path, "tictactoe_model.onnx")
    
    torch.onnx.export(
        model,                         # 변환할 모델 객체
        dummy_input,                   # 모델의 입력 텐서
        onnx_file_path,                # 저장될 ONNX 파일 경로
        export_params=True,            # 모델 내의 학습된 가중치를 포함할지 여부
        opset_version=14,              # ONNX 지원 버전
        do_constant_folding=True,      # 최적화: 상수 폴딩 활성화
        input_names=['input_board'],   # Javascript/웹에서 모델에 데이터를 넣을 때 쓸 입력 이름
        output_names=['output_logits'],# 예측 결과를 받을 때 쓸 출력 이름
        dynamic_axes={                 # 배치 사이즈를 가변적으로 열어둠
            'input_board': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )
    
    print(f"변환됨 | 경로->\n {onnx_file_path}")

if __name__ == "__main__":
    convert_to_onnx()
