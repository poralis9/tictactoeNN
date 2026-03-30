import torch



WIN_MATRIX = torch.tensor([
    [1, 0, 0,  1, 0, 0,  1, 0], # 0번 칸
    [1, 0, 0,  0, 1, 0,  0, 0], # 1번 칸
    [1, 0, 0,  0, 0, 1,  0, 1], # 2번 칸
    [0, 1, 0,  1, 0, 0,  0, 0], # 3번 칸
    [0, 1, 0,  0, 1, 0,  1, 1], # 4번 칸
    [0, 1, 0,  0, 0, 1,  0, 0], # 5번 칸
    [0, 0, 1,  1, 0, 0,  0, 1], # 6번 칸
    [0, 0, 1,  0, 1, 0,  0, 0], # 7번 칸
    [0, 0, 1,  0, 0, 1,  1, 0]  # 8번 칸
], dtype=torch.float32)

def create_batch_boards(batch_size):
    # (N, 3, 3) 형태로 한꺼번에 N개의 보드를 만듭니다.
    return torch.zeros((batch_size, 3, 3), dtype=torch.float32)

def check_winner_parallel(boards):
    '''
    boards : (batch,x+y)
    '''
    batch_size = boards.shape[0]
    flat_boards = boards.view(batch_size, 9)
    all_sums = flat_boards @ WIN_MATRIX 
    
    # 결과 저장용 (N,)
    results = torch.full((batch_size,), 2, dtype=torch.float32)
    
    # 각 줄의 합이 3이면 1번 플레이어 승리, -3이면 2번 플레이어 승리
    results[(all_sums == 3).any(dim=1)] = 1.0
    results[(all_sums == -3).any(dim=1)] = -1.0
    
    # 승자가 없는데 빈칸(0)도 없는 보드는 무승부(0) 처리
    no_winner = (results == 2)
    no_empty_space = (flat_boards != 0).all(dim=1)
    results[no_winner & no_empty_space] = 0.0
    
    return results
test_board = torch.tensor([
    [ 1.,  0.,  1.],
    [-1.,  -1., -1.],
    [ 0.,  0.,  0.]
])
