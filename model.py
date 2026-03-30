import torch
import torch.nn as nn
import torch.nn.functional as F



class TicTacToeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = x.view(-1, 9) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x) 
        
        return logits
def get_batch_actions(logits, boards):
    """
    logits: (N, 9), boards: (N, 3, 3)
    N개 게임의 최적 행동을 한 번에 계산
    """
    flat_boards = boards.view(boards.shape[0], -1)
    mask = (flat_boards != 0) # (N, 9)
    
    # 이미 돌이 있는 자리는 -1e9로 덮어쓰기
    masked_logits = logits.masked_fill(mask, -1e9)
    
    # 각 행(게임)별로 가장 높은 점수의 인덱스 추출
    actions = torch.argmax(masked_logits, dim=1) # (N,)
    
    return actions
if __name__ =="main":

    model = TicTacToeNN()
    test_board = torch.zeros((1, 3, 3), dtype=torch.float32)
    output = model(test_board)

    print("신경망 출력값 (9개 칸에 대한 점수):")
    print(output)