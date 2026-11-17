import cv2
import numpy as np
from scipy import signal
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os

app = Flask(__name__, static_folder='.')
CORS(app)

# --- Global State & Buffers ---
class RPPGState:
    def __init__(self):
        self.reset()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.fps = 10.0  # Estimated frame rate based on JS interval

    def reset(self):
        self.is_measuring = False
        self.raw_rgb = []
        self.timestamps = []
        self.hr = None
        self.rr = None
        self.hrv = None
        self.signal_quality = 0.0
        self.face_detected = False
        self.lighting_quality = 0.0
        self.processing_status = 'Idle'
        self.current_signal_sample = 0.0
        self.start_time = None

state = RPPGState()

# --- Signal Processing Algorithms ---
def extract_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = state.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        state.face_detected = False
        return None
    
    state.face_detected = True
    (x, y, w, h) = faces[0]
    
    # Forehead ROI (Top 30% of face)
    forehead_y1 = y
    forehead_y2 = y + int(h * 0.3)
    forehead_x1 = x + int(w * 0.2)
    forehead_x2 = x + int(w * 0.8)
    
    # Cheeks ROI (Left and Right 20%)
    cheek_y1 = y + int(h * 0.4)
    cheek_y2 = y + int(h * 0.7)
    cheek_left_x1 = x
    cheek_left_x2 = x + int(w * 0.2)
    cheek_right_x1 = x + int(w * 0.8)
    cheek_right_x2 = x + w

    rois = [
        frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2],
        frame[cheek_y1:cheek_y2, cheek_left_x1:cheek_left_x2],
        frame[cheek_y1:cheek_y2, cheek_right_x1:cheek_right_x2]
    ]
    
    valid_rois = [r for r in rois if r.size > 0]
    if not valid_rois:
        return None
        
    return np.mean([np.mean(r, axis=(0, 1)) for r in valid_rois], axis=0)

def pos_algorithm(rgb_signals):
    """Plane-Orthogonal-to-Skin (POS) algorithm for motion robustness."""
    R, G, B = rgb_signals[:, 0], rgb_signals[:, 1], rgb_signals[:, 2]
    
    # Normalize
    R_n = R / np.mean(R)
    G_n = G / np.mean(G)
    B_n = B / np.mean(B)
    
    X = 3 * R_n - 2 * G_n
    Y = 1.5 * R_n + G_n - 1.5 * B_n
    
    # Sliding window projection
    alpha = np.std(X) / np.std(Y) if np.std(Y) > 0 else 1.0
    S = X - alpha * Y
    return S

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        return data
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def estimate_hr(ppg_signal, fs):
    # HR bandpass: 0.7 Hz to 4.0 Hz (42 to 240 BPM)
    filtered = bandpass_filter(ppg_signal, 0.7, 4.0, fs)
    
    # Apply Hanning window to reduce spectral leakage
    windowed = filtered * np.hanning(len(filtered))
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / fs)
    
    # Find peak in valid HR range
    valid_idx = np.where((freqs >= 0.7) & (freqs <= 4.0))[0]
    if len(valid_idx) == 0:
        return None, 0.0
    
    peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
    hr_bpm = freqs[peak_idx] * 60.0
    
    # Signal Quality (SNR approximation)
    signal_power = np.max(fft_vals[valid_idx])
    noise_power = np.mean(fft_vals[valid_idx])
    snr = signal_power / (noise_power + 1e-6)
    quality = min(snr / 10.0, 1.0) # Normalize to 0-1
    
    return hr_bpm, quality

def estimate_hrv(ppg_signal, fs):
    # Filter for peak detection
    filtered = bandpass_filter(ppg_signal, 0.7, 4.0, fs)
    
    # Find peaks (pulses)
    # Distance is minimum time between beats in samples (e.g., 0.4s = 240 BPM max)
    min_dist = int(0.4 * fs) 
    peaks, _ = signal.find_peaks(filtered, distance=min_dist, prominence=0.1)
    
    if len(peaks) < 3:
        return None
        
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(peaks) / fs * 1000.0
    
    # RMSSD
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return rmssd

def estimate_rr(ppg_signal, fs):
    # Respiratory rate is often found in the low frequency modulation of the pulse signal
    # Bandpass: 0.1 Hz to 0.5 Hz (6 to 30 breaths/min)
    filtered = bandpass_filter(ppg_signal, 0.1, 0.5, fs)
    
    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / fs)
    
    valid_idx = np.where((freqs >= 0.1) & (freqs <= 0.5))[0]
    if len(valid_idx) == 0:
        return None
        
    peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
    rr_bpm = freqs[peak_idx] * 60.0
    
    return rr_bpm

# --- Flask Endpoints ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/start', methods=['GET', 'POST'])
def start_measurement():
    state.is_measuring = True
    state.processing_status = 'Collecting Signal'
    state.start_time = time.time()
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_measurement():
    state.is_measuring = False
    state.processing_status = 'Idle'
    return jsonify({'status': 'stopped'})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_measuring': state.is_measuring,
        'face_detected': state.face_detected,
        'hr': state.hr,
        'rr': state.rr,
        'hrv': state.hrv,
        'signal_quality': state.signal_quality,
        'processing_status': state.processing_status
    })

@app.route('/api/frame', methods=['POST'])
def process_frame():
    if not state.is_measuring:
        return jsonify({'error': 'Not measuring'}), 400

    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'No frame'}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        # 1. Extract ROI and RGB
        rgb_mean = extract_roi(frame)
        
        if rgb_mean is None:
            state.lighting_quality = 0.0
            return jsonify(get_response_payload())

        # Lighting quality based on overall brightness
        brightness = np.mean(rgb_mean) / 255.0
        state.lighting_quality = 1.0 - abs(brightness - 0.5) * 2 # Best at 0.5 (mid-gray)

        # 2. Buffer Management (Rolling buffer of ~60 seconds)
        max_samples = int(60 * state.fps)
        state.raw_rgb.append(rgb_mean)
        state.timestamps.append(time.time())
        
        if len(state.raw_rgb) > max_samples:
            state.raw_rgb.pop(0)
            state.timestamps.pop(0)

        # 3. Signal Processing (Only if we have enough data, e.g., 10 seconds)
        if len(state.raw_rgb) >= int(10 * state.fps):
            state.processing_status = 'Processing Signal'
            rgb_array = np.array(state.raw_rgb)
            
            # Extract pulse signal using POS
            ppg_signal = pos_algorithm(rgb_array)
            
            # Update live waveform sample (normalized)
            state.current_signal_sample = (ppg_signal[-1] - np.mean(ppg_signal)) / (np.std(ppg_signal) + 1e-6)
            
            # Estimate parameters
            hr, quality = estimate_hr(ppg_signal, state.fps)
            state.signal_quality = quality
            
            if quality > 0.4: # Only trust HR if quality is acceptable
                state.hr = hr
                state.hrv = estimate_hrv(ppg_signal, state.fps)
                state.rr = estimate_rr(ppg_signal, state.fps)
                state.processing_status = 'Good Signal'
            else:
                state.processing_status = 'Poor Signal'
        else:
            state.processing_status = 'Collecting Signal'

        return jsonify(get_response_payload())

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_response_payload():
    return {
        'face_detected': state.face_detected,
        'lighting_quality': float(state.lighting_quality),
        'signal_quality': float(state.signal_quality),
        'processing_status': state.processing_status,
        'hr': float(state.hr) if state.hr is not None else None,
        'rr': float(state.rr) if state.rr is not None else None,
        'hrv': float(state.hrv) if state.hrv is not None else None,
        'signal_sample': float(state.current_signal_sample)
    }

if __name__ == '__main__':
    # Run with threaded=True to handle concurrent frame uploads and status checks
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)