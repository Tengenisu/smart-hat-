from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import threading
import time
import json
import os
from detect import detect_frame, set_audio_enabled, set_detection_threshold, get_current_detections

app = Flask(__name__)
# Disable debug mode on production to improve performance
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
camera_lock = threading.Lock()
audio_enabled = True
detection_threshold = 0.5
emergency_contact = "911"  # Default emergency contact
camera = None  # Camera object - defined globally for proper cleanup

# Configure for Raspberry Pi Camera if available
USING_PI_CAMERA = False
try:
    from picamera2 import Picamera2
    USING_PI_CAMERA = True
    print("Using Raspberry Pi Camera")
except ImportError:
    print("Picamera2 not available, falling back to regular OpenCV camera")

# Thread for sending detection updates to the client
detection_thread = None
detection_thread_running = False

def detection_update_thread():
    """Thread that sends detection updates to connected clients"""
    global detection_thread_running
    while detection_thread_running:
        try:
            # Get current detections from detect.py
            detections = get_current_detections()
            if detections:
                # Send detections to all connected clients
                socketio.emit('detection_update', detections)
        except Exception as e:
            print(f"Error in detection thread: {e}")
        time.sleep(1.0)  # Reduced update frequency to 1 second for Pi

def init_camera():
    """Initialize the camera based on what's available"""
    global USING_PI_CAMERA
    
    if USING_PI_CAMERA:
        # Use the Raspberry Pi camera
        try:
            picam = Picamera2()
            config = picam.create_preview_configuration(main={"size": (640, 480)})
            picam.configure(config)
            picam.start()
            time.sleep(1)  # Give camera time to start
            return picam
        except Exception as e:
            print(f"Error initializing Pi Camera: {e}")
            USING_PI_CAMERA = False
            # Fall back to USB camera
    
    # Use regular OpenCV camera (USB webcam)
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not access the webcam")
            
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None

def read_frame(cam):
    """Read a frame from either Pi Camera or regular camera"""
    if USING_PI_CAMERA:
        # Pi Camera
        return True, cam.capture_array()
    else:
        # Regular OpenCV camera
        return cam.read()

def gen_frames():
    global camera
    camera = init_camera()
    
    if camera is None:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() +
               b'\r\n')
        return

    try:
        while True:
            with camera_lock:
                success, frame = read_frame(camera)
                if not success:
                    break

                # Apply object detection with current settings
                try:
                    frame = detect_frame(frame, audio_enabled, detection_threshold)
                except Exception as e:
                    print(f"Error in detect_frame: {e}")
                    # Draw error message on frame
                    cv2.putText(frame, f"Detection error", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

                # Encode frame to JPEG
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Add a slight delay to reduce CPU usage
            time.sleep(0.05)

    finally:
        # Release camera resources
        if camera is not None:
            if not USING_PI_CAMERA:
                camera.release()
            else:
                camera.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/enable_audio', methods=['POST'])
def enable_audio():
    global audio_enabled
    audio_enabled = True
    set_audio_enabled(True)
    socketio.emit('audio_status', {'enabled': True})
    return jsonify({"status": "success"})

@app.route('/disable_audio', methods=['POST'])
def disable_audio():
    global audio_enabled
    audio_enabled = False
    set_audio_enabled(False)
    socketio.emit('audio_status', {'enabled': False})
    return jsonify({"status": "success"})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global detection_threshold
    data = request.json
    threshold = float(data.get('threshold', 50)) / 100  # Convert from 0-100 to 0-1
    detection_threshold = threshold
    set_detection_threshold(threshold)
    return jsonify({"status": "success", "threshold": threshold})

@app.route('/set_emergency_contact', methods=['POST'])
def set_emergency_contact():
    global emergency_contact
    data = request.json
    emergency_contact = data.get('contact', '911')
    return jsonify({"status": "success", "contact": emergency_contact})

@app.route('/trigger_emergency', methods=['POST'])
def trigger_emergency():
    # In a real application, this would send an alert to the emergency contact
    # For now, we'll just log it and notify all connected clients
    socketio.emit('emergency_triggered', {'contact': emergency_contact, 'time': time.strftime('%H:%M:%S')})
    return jsonify({"status": "success", "message": f"Emergency alert sent to {emergency_contact}"})

@app.route('/system_info', methods=['GET'])
def system_info():
    """Return Raspberry Pi system information"""
    cpu_temp = "Unknown"
    cpu_usage = "Unknown"
    ram_usage = "Unknown"
    
    # Get CPU temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000
            cpu_temp = f"{temp:.1f}°C"
    except:
        pass
    
    # Get CPU usage
    try:
        import psutil
        cpu_usage = f"{psutil.cpu_percent()}%"
        ram = psutil.virtual_memory()
        ram_usage = f"{ram.percent}%"
    except:
        pass
    
    return jsonify({
        "cpu_temp": cpu_temp,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage
    })

@app.errorhandler(500)
def handle_error(e):
    return render_template('error.html', error=str(e)), 500

@socketio.on('connect')
def handle_connect():
    # Send initial status to newly connected client
    emit('audio_status', {'enabled': audio_enabled})
    emit('threshold_status', {'threshold': int(detection_threshold * 100)})

@socketio.on('disconnect')
def handle_disconnect():
    pass

def start_detection_thread():
    global detection_thread, detection_thread_running
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread_running = True
        detection_thread = threading.Thread(target=detection_update_thread)
        detection_thread.daemon = True
        detection_thread.start()

if __name__ == '__main__':
    # Import numpy here to avoid circular import
    import numpy as np
    import RPi.GPIO as GPIO
    
    try:
        start_detection_thread()
        # Use default Flask server instead of debug mode for better performance
        # Change host to '0.0.0.0' to allow access from other devices in the network
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("Shutting down...")
        detection_thread_running = False
        if detection_thread:
            detection_thread.join(timeout=1.0)
        # Cleanup camera if needed
        if camera is not None and not USING_PI_CAMERA:
            camera.release()
        # Clean up GPIO resources
        GPIO.cleanup()
