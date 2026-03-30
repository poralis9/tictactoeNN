// 상태 변수
let boardState = new Array(9).fill(0); // 0: 빈칸, 1: 1p, -1: 2p
let session = null;
let humanPlayer = 1; // 사람이 1(선공)인지 -1(후공)인지
let aiPlayer = -1;
let currentPlayer = 1; // 현재 턴 (무조건 1부터 시작)
let isGameOver = false;

// DOM 요소
const cells = document.querySelectorAll('.cell');
const statusText = document.getElementById('status');
const btnFirst = document.getElementById('btn-first');
const btnSecond = document.getElementById('btn-second');
const btnReset = document.getElementById('btn-reset');

// 승리 조합 (가로, 세로, 대각선)
const winPatterns = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // 가로
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // 세로
    [0, 4, 8], [2, 4, 6]             // 대각선
];

// 1. 모델 로드
async function loadModel() {
    try {
        // export_onnx.py로 뽑아낸 모델 로드
        session = await ort.InferenceSession.create('./tictactoe_model.onnx');
        statusText.innerText = "game ready your turn now.";
        btnReset.disabled = false;
    } catch (e) {
        console.error(e);
        statusText.innerText = "can`t load model.";
    }
}

// 2. 승패 판정 로직 (board.py의 check_winner_parallel 동일 구현)
function checkWinner() {
    for (let pattern of winPatterns) {
        const sum = boardState[pattern[0]] + boardState[pattern[1]] + boardState[pattern[2]];
        if (sum === 3) return 1;    // 1번 플레이어 승리
        if (sum === -3) return -1;  // -1번 플레이어 승리
    }
    if (!boardState.includes(0)) return 0; // 무승부
    return null; // 진행 중
}

// 3. 화면 업데이트
function updateBoard() {
    cells.forEach((cell, index) => {
        const val = boardState[index];
        cell.innerText = val === 1 ? (humanPlayer === 1 ? 'O' : 'X') : 
                         val === -1 ? (humanPlayer === -1 ? 'O' : 'X') : '';
        cell.className = 'cell ' + (val === 1 ? (humanPlayer === 1 ? 'o' : 'x') : 
                                   val === -1 ? (humanPlayer === -1 ? 'o' : 'x') : '');
    });
}

// 4. 게임 상태 확인 및 처리
function handleGameState() {
    const winner = checkWinner();
    if (winner !== null) {
        isGameOver = true;
        if (winner === 0) statusText.innerText = "draw";
        else if (winner === humanPlayer) statusText.innerText = "you win";
        else statusText.innerText = "you lost";
        return true;
    }
    return false;
}

// 5. AI 추론 (파이썬 로직 포팅)
async function makeAiMove() {
    if (isGameOver || !session) return;
    
    statusText.innerText = "thinking...";
    
    // 파이썬 로직: input_boards = current_boards * current_players
    // 즉, AI가 볼 때는 자신의 돌이 무조건 1, 상대가 -1이 되어야 함
    const inputData = Float32Array.from(boardState.map(val => val * aiPlayer));
    
    // 텐서 생성 (배치1, 3x3)
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 3]);
    
    // 추론 실행 (입력명 input_board)
    const results = await session.run({ input_board: tensor });
    
    // 출력 로짓 (출력명 output_logits)
    const logits = results.output_logits.data;
    
    // 이미 돌이 있는 자리는 마스킹 (-1e9) 처리 후 argmax
    let bestScore = -Infinity;
    let bestMove = -1;
    
    for (let i = 0; i < 9; i++) {
        if (boardState[i] === 0) { // 빈 칸만 확인
            if (logits[i] > bestScore) {
                bestScore = logits[i];
                bestMove = i;
            }
        }
    }
    
    // 보드 업데이트 및 턴 넘기기
    if (bestMove !== -1) {
        boardState[bestMove] = aiPlayer;
        currentPlayer = humanPlayer;
        updateBoard();
        if (!handleGameState()) {
            statusText.innerText = "your turn";
        }
    }
}

// 6. 사람의 클릭 이벤트
cells.forEach(cell => {
    cell.addEventListener('click', async (e) => {
        const idx = parseInt(e.target.dataset.index);
        
        // 게임 오버 상태이거나, 이미 돌이 있거나, AI 턴이면 무시
        if (isGameOver || boardState[idx] !== 0 || currentPlayer !== humanPlayer) return;

        // 보드 업데이트
        boardState[idx] = humanPlayer;
        currentPlayer = aiPlayer;
        updateBoard();
        
        if (!handleGameState()) {
            // AI 턴 호출 (살짝 지연시켜서 자연스럽게)
            setTimeout(makeAiMove, 100);
        }
    });
});

// 7. 게임 초기화
function resetGame(isHumanFirst) {
    boardState.fill(0);
    isGameOver = false;
    
    if (isHumanFirst) {
        humanPlayer = 1;
        aiPlayer = -1;
        currentPlayer = 1;
        btnFirst.classList.add('active');
        btnSecond.classList.remove('active');
        statusText.innerText = "your turn";
    } else {
        humanPlayer = -1;
        aiPlayer = 1;
        currentPlayer = 1; // 1부터 시작하므로 AI가 1이 됨
        btnFirst.classList.remove('active');
        btnSecond.classList.add('active');
        makeAiMove(); // AI 먼저 시작
    }
    updateBoard();
}

btnFirst.addEventListener('click', () => resetGame(true));
btnSecond.addEventListener('click', () => resetGame(false));
btnReset.addEventListener('click', () => resetGame(humanPlayer === 1));

// 시작 시 모델 로드
loadModel();
