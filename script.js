// PulseSight Frontend Controller
const API_BASE = ''; 
const video = document.getElementById('video-feed');
const procCanvas = document.getElementById('processing-canvas');
const procCtx = procCanvas.getContext('2d', { willReadFrequently: true });
const waveCanvas = document.getElementById('waveform-canvas');
const waveCtx = waveCanvas.getContext('2d');

// UI Elements
const ui = {
    connStatus: document.getElementById('camera-connection-status'),
    faceStatus: document.getElementById('face-status'),
    lightStatus: document.getElementById('lighting-status'),
    timer: document.getElementById('measurement-timer'),
    hr: document.getElementById('hr-value'),
    rr: document.getElementById('rr-value'),
    hrv: document.getElementById('hrv-value'),
    sigQuality: document.getElementById('signal-quality'),
    procStatus: document.getElementById('processing-status'),
    progressBar: document.getElementById('progress-bar'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    btnRecalc: document.getElementById('btn-recalculate'),
    btnReset: document.getElementById('btn-reset')
};

let isMeasuring = false;
let isSendingFrame = false;
let stream = null;
let frameInterval = null;
let timerInterval = null;
let startTime = 0;
let waveformHistory = new Array(100).fill(0);

// 1. Camera Initialization
async function initCamera() {
    try {
        ui.connStatus.textContent = 'Initializing Camera...';
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, 
            audio: false 
        });
        video.srcObject = stream;
        await video.play();
        ui.connStatus.textContent = 'Camera Active';
        ui.connStatus.style.color = 'var(--success)';
    } catch (err) {
        ui.connStatus.textContent = 'Camera Access Denied';
        ui.connStatus.style.color = 'var(--danger)';
        console.error('Camera error:', err);
    }
}

// 2. Frame Capture & Transmission
function startFrameLoop() {
    procCanvas.width = 320;
    procCanvas.height = 240;
    
    frameInterval = setInterval(async () => {
        if (!isMeasuring || isSendingFrame) return;
        isSendingFrame = true;

        try {
            procCtx.drawImage(video, 0, 0, procCanvas.width, procCanvas.height);
            
            procCanvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');
                
                const res = await fetch(`${API_BASE}/api/frame`, {
                    method: 'POST',
                    body: formData
                });
                
                if (res.ok) {
                    const data = await res.json();
                    updateUI(data);
                }
                isSendingFrame = false;
            }, 'image/jpeg', 0.7); // 70% quality for performance
        } catch (e) {
            isSendingFrame = false;
        }
    }, 100); // 10 FPS max to prevent backend overload
}

// 3. UI Updates
function updateUI(data) {
    ui.faceStatus.textContent = data.face_detected ? 'Face Locked' : 'No Face';
    ui.faceStatus.style.color = data.face_detected ? 'var(--success)' : 'var(--danger)';
    
    ui.lightStatus.textContent = data.lighting_quality > 0.6 ? 'Good' : (data.lighting_quality > 0.3 ? 'Fair' : 'Poor');
    ui.sigQuality.textContent = data.signal_quality > 0.7 ? 'Excellent' : (data.signal_quality > 0.4 ? 'Acceptable' : 'Poor');
    ui.procStatus.textContent = data.processing_status || 'Processing';

    if (data.hr !== null) { ui.hr.textContent = Math.round(data.hr); ui.hr.parentElement.classList.add('active'); }
    if (data.rr !== null) { ui.rr.textContent = Math.round(data.rr); ui.rr.parentElement.classList.add('active'); }
    if (data.hrv !== null) { ui.hrv.textContent = Math.round(data.hrv); ui.hrv.parentElement.classList.add('active'); }

    // Update Waveform
    if (data.signal_sample !== undefined) {
        waveformHistory.shift();
        waveformHistory.push(data.signal_sample);
        drawWaveform();
    }

    // Progress (assume 60s is full measurement)
    const elapsed = (Date.now() - startTime) / 1000;
    const progress = Math.min((elapsed / 60) * 100, 100);
    ui.progressBar.style.width = `${progress}%`;
}

function drawWaveform() {
    const w = waveCanvas.width = waveCanvas.offsetWidth;
    const h = waveCanvas.height = waveCanvas.offsetHeight;
    waveCtx.clearRect(0, 0, w, h);
    
    waveCtx.beginPath();
    waveCtx.strokeStyle = '#35D9FF';
    waveCtx.lineWidth = 2;
    waveCtx.shadowBlur = 5;
    waveCtx.shadowColor = '#35D9FF';

    const step = w / (waveformHistory.length - 1);
    const mid = h / 2;
    
    waveformHistory.forEach((val, i) => {
        const x = i * step;
        const y = mid - (val * mid * 0.8); 
        if (i === 0) waveCtx.moveTo(x, y);
        else waveCtx.lineTo(x, y);
    });
    waveCtx.stroke();
}

// 4. Timer
function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        ui.timer.textContent = `${mins}:${secs}`;
    }, 1000);
}

// 5. Controls
ui.btnStart.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/start`, { method: 'POST' });
    isMeasuring = true;
    ui.btnStart.disabled = true;
    ui.btnStop.disabled = false;
    ui.btnRecalc.disabled = false;
    startFrameLoop();
    startTimer();
});

ui.btnStop.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
    isMeasuring = false;
    clearInterval(frameInterval);
    clearInterval(timerInterval);
    ui.btnStart.disabled = false;
    ui.btnStop.disabled = true;
});

ui.btnRecalc.addEventListener('click', async () => {
    ui.procStatus.textContent = 'Recalculating...';
    // Triggers a fresh calculation on the backend without resetting the buffer
    await fetch(`${API_BASE}/api/start`, { method: 'POST' }); 
});

ui.btnReset.addEventListener('click', async () => {
    await fetch(`${API_BASE}/api/stop`, { method: 'POST' });
    isMeasuring = false;
    clearInterval(frameInterval);
    clearInterval(timerInterval);
    
    ui.timer.textContent = '00:00';
    ui.progressBar.style.width = '0%';
    ui.hr.textContent = '--'; ui.rr.textContent = '--'; ui.hrv.textContent = '--';
    document.querySelectorAll('.param-card').forEach(c => c.classList.remove('active'));
    waveformHistory = new Array(100).fill(0);
    drawWaveform();
    
    ui.btnStart.disabled = false;
    ui.btnStop.disabled = true;
    ui.btnRecalc.disabled = true;
});

// Initialize
window.addEventListener('resize', drawWaveform);
initCamera();