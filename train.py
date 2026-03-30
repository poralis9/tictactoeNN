import torch
import torch.nn as nn
import torch.optim as optim
import random
from model import TicTacToeNN, get_batch_actions
from board import create_batch_boards, check_winner_parallel
from collections import deque
import math
from tqdm import tqdm
model = TicTacToeNN()
target_model = TicTacToeNN() 
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.SmoothL1Loss()

def train_from_memory(memory, batch_size, gamma):
    if len(memory) < batch_size:
        return None
    batch = random.sample(memory, batch_size)
    states = torch.stack([s for s, a, r, ns, d in batch])
    actions = torch.tensor([a for s, a, r, ns, d in batch])
    rewards = torch.tensor([r for s, a, r, ns, d in batch])
    next_states = torch.stack([ns if ns is not None else torch.zeros((3, 3)) 
                               for s, a, r, ns, d in batch])
    dones = torch.tensor([d for s, a, r, ns, d in batch], dtype=torch.float32)
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_logits = target_model(next_states)
    
        flat_next_states = next_states.view(batch_size, -1)
        next_mask = (flat_next_states != 0)
        next_logits = next_logits.masked_fill(next_mask, -1e9)
        max_next_q = next_logits.max(dim=1)[0]
    target_q = rewards + gamma * max_next_q * (1 - dones)
    current_logits = model(states)
    current_q = current_logits.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = criterion(current_q, target_q)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()





#config
NUM_ENVS = 1000       
MEMORY_SIZE = 50000   
BATCH_SIZE = 64       
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 2000
TRAIN_STEPS = 150
memory = deque(maxlen=MEMORY_SIZE)
def play_parallel_games_and_train(epoch):
    current_epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * epoch / EPS_DECAY)
    boards = create_batch_boards(NUM_ENVS)
    players = torch.ones(NUM_ENVS, dtype=torch.float32)
    histories = [[] for _ in range(NUM_ENVS)]
    active_games = torch.ones(NUM_ENVS, dtype=torch.bool)
    while active_games.any():
        active_idx = active_games.nonzero().flatten()
        current_boards = boards[active_idx]
        current_players = players[active_idx]
        input_boards = current_boards * current_players.view(-1, 1, 1)
        with torch.no_grad():
            logits = model(input_boards)
            best_actions = get_batch_actions(logits, current_boards)
            
        actions = []
        random_mask = torch.rand(len(active_idx)) < current_epsilon
        flat_boards = current_boards.view(len(active_idx), -1)
        valid_mask = (flat_boards == 0)
        rand_logits = torch.rand_like(flat_boards)
        rand_logits.masked_fill_(~valid_mask, -1e9)
        rand_actions = torch.argmax(rand_logits, dim=1)
        actions = torch.where(random_mask, rand_actions, best_actions)
        for i, env_idx in enumerate(active_idx):
            action = actions[i]
            player = current_players[i].item()
            histories[env_idx].append((input_boards[i].clone(), action, player))
            boards[env_idx].view(-1)[action] = player
        statuses = check_winner_parallel(boards[active_idx])
        for i, env_idx in enumerate(active_idx):
            status = statuses[i].item()
            
            if status != 2:
                active_games[env_idx] = False
                for j, (state, action, player) in enumerate(histories[env_idx]):
                    next_state_idx = j + 2
                    if next_state_idx >= len(histories[env_idx]):
                        if status == 0: reward = 0.5        
                        elif status == player: reward = 1.0 
                        else: reward = -1.0                 
                        next_state = None
                        done = True
                    
                    else:
                        reward = 0.0
                        next_state = histories[env_idx][next_state_idx][0]
                        done = False
                    memory.append((state, action, reward, next_state, done))
                    
        players[active_idx] *= -1
    total_loss = 0.0
    valid_updates = 0
    for _ in range(TRAIN_STEPS):
        loss = train_from_memory(memory, BATCH_SIZE, GAMMA)
        if loss is not None:
            total_loss += loss
            valid_updates += 1
    target_model.load_state_dict(model.state_dict())        
    if valid_updates > 0:
        return total_loss / valid_updates, current_epsilon
    return 0.0, current_epsilon


if __name__ == "__main__":
    epochs = 4000
    pbar = tqdm(range(epochs), desc="Training TicTacToe AI", mininterval=1.0)
    for epoch in pbar:
        avg_loss, curr_eps = play_parallel_games_and_train(epoch)
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Epsilon": f"{curr_eps:.3f}", "Memory": len(memory)})
        
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_path,f"tictactoe_model_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    
    print("-" * 50)
