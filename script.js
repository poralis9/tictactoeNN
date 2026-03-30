
let boardState = new Array(9).fill(0); 
let session = null;
let humanPlayer = 1; 
let aiPlayer = -1;
let currentPlayer = 1; 
let isGameOver = false;

const cells = document.querySelectorAll('.cell');
const statusText = document.getElementById('status');
const btnFirst = document.getElementById('btn-first');
const btnSecond = document.getElementById('btn-second');
const btnReset = document.getElementById('btn-reset');


const winPatterns = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], 
    [0, 3, 6], [1, 4, 7], [2, 5, 8], 
    [0, 4, 8], [2, 4, 6]             
];

async function loadModel() {
    try {
        session = await ort.InferenceSession.create('./tictactoe_model.onnx');
        statusText.innerText = "game ready your turn now.";
        btnReset.disabled = false;
    } catch (e) {
        console.error(e);
        statusText.innerText = "can`t load model.";
    }
}

function checkWinner() {
    for (let pattern of winPatterns) {
        const sum = boardState[pattern[0]] + boardState[pattern[1]] + boardState[pattern[2]];
        if (sum === 3) return 1;    
        if (sum === -3) return -1;  
    }
    if (!boardState.includes(0)) return 0; 
    return null; 
}

function updateBoard() {
    cells.forEach((cell, index) => {
        const val = boardState[index];
        cell.innerText = val === 1 ? (humanPlayer === 1 ? 'O' : 'X') : 
                         val === -1 ? (humanPlayer === -1 ? 'O' : 'X') : '';
        cell.className = 'cell ' + (val === 1 ? (humanPlayer === 1 ? 'o' : 'x') : 
                                   val === -1 ? (humanPlayer === -1 ? 'o' : 'x') : '');
    });
}

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

async function makeAiMove() {
    if (isGameOver || !session) return;
    
    statusText.innerText = "thinking...";
    
    const inputData = Float32Array.from(boardState.map(val => val * aiPlayer));
    
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 3]);
    
    const results = await session.run({ input_board: tensor });
    
    const logits = results.output_logits.data;
    
    let bestScore = -Infinity;
    let bestMove = -1;
    
    for (let i = 0; i < 9; i++) {
        if (boardState[i] === 0) {
            if (logits[i] > bestScore) {
                bestScore = logits[i];
                bestMove = i;
            }
        }
    }
    
    if (bestMove !== -1) {
        boardState[bestMove] = aiPlayer;
        currentPlayer = humanPlayer;
        updateBoard();
        if (!handleGameState()) {
            statusText.innerText = "your turn";
        }
    }
}

cells.forEach(cell => {
    cell.addEventListener('click', async (e) => {
        const idx = parseInt(e.target.dataset.index);
        
        if (isGameOver || boardState[idx] !== 0 || currentPlayer !== humanPlayer) return;
        boardState[idx] = humanPlayer;
        currentPlayer = aiPlayer;
        updateBoard();
        
        if (!handleGameState()) {
            setTimeout(makeAiMove, 100);
        }
    });
});

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
        currentPlayer = 1;
        btnFirst.classList.remove('active');
        btnSecond.classList.add('active');
        makeAiMove(); 
    }
    updateBoard();
}

btnFirst.addEventListener('click', () => resetGame(true));
btnSecond.addEventListener('click', () => resetGame(false));
btnReset.addEventListener('click', () => resetGame(humanPlayer === 1));

loadModel();
