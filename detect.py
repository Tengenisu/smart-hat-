import torch
import cv2
import pyttsx3
import time
import threading
import numpy as np
import RPi.GPIO as GPIO
import smbus2

# Set up GPIO for ultrasonic sensor
# Using HC-SR04 as default - modify pins and code for your specific sensor
GPIO.setmode(GPIO.BCM)
TRIG_PIN = 23  # GPIO pin for TRIGGER
ECHO_PIN = 24  # GPIO pin for ECHO
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Initialize I2C bus for VL53L0X ToF sensor (if using instead of ultrasonic)
try:
    bus = smbus2.SMBus(1)  # For I2C devices
    TOF_SENSOR_AVAILABLE = True
    print("I2C bus initialized for ToF sensor")
except:
    TOF_SENSOR_AVAILABLE = False
    print("No I2C bus available, ToF sensor will not be used")

# Choose your distance sensor type
DISTANCE_SENSOR_TYPE = "ULTRASONIC"  # Options: "ULTRASONIC", "TOF", "NONE"

# Load a lighter model - YOLOv5n (nano) instead of YOLOv5s
# You can also try using a pre-downloaded model to avoid internet dependency
try:
    # Try loading from local file first (if you've downloaded it)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', force_reload=False)
except:
    # Fall back to downloading - note this requires internet on first run
    print("Downloading YOLOv5n model - this may take a moment...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu')

# Optimize model for inference
model.eval()
if torch.cuda.is_available():  # Not likely on Pi, but good practice
    model.half()  # Use FP16 if available
else:
    # Force model to use CPU
    model = model.to('cpu')

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed

# Global variables
audio_enabled = True
detection_threshold = 0.5
last_spoken = {}  # Track when objects were last announced
speak_cooldown = 5  # Seconds between repeating the same object
speaking_lock = threading.Lock()
current_detections = []  # Store current detections for WebSocket updates
process_every_n_frames = 3  # Only process every nth frame to reduce CPU load
frame_counter = 0
max_width = 640  # Limit frame size for processing
current_distance = 0  # Current measured distance in cm

def measure_distance_ultrasonic():
    """Measure distance using HC-SR04 ultrasonic sensor"""
    try:
        # Send 10us pulse to trigger
        GPIO.output(TRIG_PIN, True)
        time.sleep(0.00001)
        GPIO.output(TRIG_PIN, False)
        
        # Wait for echo to start
        pulse_start = time.time()
        timeout = pulse_start + 0.1  # 100ms timeout
        while GPIO.input(ECHO_PIN) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None  # Timeout - no echo received
        
        # Wait for echo to end
        pulse_end = time.time()
        timeout = pulse_end + 0.1  # 100ms timeout
        while GPIO.input(ECHO_PIN) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None  # Timeout - echo too long
        
        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound = 343m/s = 34300cm/s / 2 (round trip)
        distance = round(distance, 2)
        
        # Filter out unreasonable values (typical HC-SR04 range: 2cm to 400cm)
        if 2 <= distance <= 400:
            return distance
        else:
            return None
            
    except Exception as e:
        print(f"Error measuring distance: {e}")
        return None

def measure_distance_tof():
    """Measure distance using VL53L0X Time-of-Flight sensor"""
    try:
        if not TOF_SENSOR_AVAILABLE:
            return None
            
        # This is a simplified implementation
        # For a real VL53L0X, you would use the appropriate library
        # such as VL53L0X_python or adafruit-circuitpython-vl53l0x
        
        # Example with adafruit-circuitpython-vl53l0x:
        # import adafruit_vl53l0x
        # i2c = board.I2C()
        # vl53 = adafruit_vl53l0x.VL53L0X(i2c)
        # distance = vl53.range
        
        # Placeholder - replace with actual reading code for your sensor
        return None
        
    except Exception as e:
        print(f"Error reading ToF sensor: {e}")
        return None

def update_distance():
    """Update the current distance reading based on selected sensor"""
    global current_distance
    
    if DISTANCE_SENSOR_TYPE == "ULTRASONIC":
        measured = measure_distance_ultrasonic()
    elif DISTANCE_SENSOR_TYPE == "TOF":
        measured = measure_distance_tof()
    else:
        return None
        
    if measured is not None:
        current_distance = measured
    
    return current_distance

def distance_monitoring_thread():
    """Thread that continuously updates distance measurements"""
    while True:
        update_distance()
        time.sleep(0.1)  # 10 Hz measurement rate

# Start the distance monitoring thread
distance_thread = threading.Thread(target=distance_monitoring_thread, daemon=True)
distance_thread.start()

def set_audio_enabled(enabled):
    global audio_enabled
    audio_enabled = enabled

def set_detection_threshold(threshold):
    global detection_threshold
    detection_threshold = threshold

def get_current_detections():
    """Return the current detections for WebSocket updates"""
    return current_detections

def speak(text):
    if not audio_enabled:
        return

    # Use a thread for speaking to avoid blocking the main thread
    def speak_thread():
        with speaking_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()

    threading.Thread(target=speak_thread).start()

def resize_if_needed(frame):
    """Resize frame if it's too large to improve performance"""
    h, w = frame.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(frame, (max_width, int(h * ratio)))
    return frame

def detect_frame(frame, audio_enabled_param=None, threshold_param=None):
    global audio_enabled, detection_threshold, current_detections, frame_counter, current_distance

    # Update global variables if parameters are provided
    if audio_enabled_param is not None:
        audio_enabled = audio_enabled_param
    if threshold_param is not None:
        detection_threshold = threshold_param

    # Create a copy of the original frame for drawing
    display_frame = frame.copy()
    
    # Get current distance from sensor
    measured_distance = current_distance  # cm
    if measured_distance > 0:
        # Convert to meters for consistency
        measured_distance = measured_distance / 100.0
    
    # Increment and check frame counter - only process some frames
    frame_counter = (frame_counter + 1) % process_every_n_frames
    
    # Skip processing this frame if it's not time
    if frame_counter != 0:
        # Display the distance even when skipping detection
        if measured_distance > 0:
            distance_text = f"Distance: {measured_distance:.2f}m"
            cv2.putText(display_frame, distance_text, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
        return display_frame
    
    # Resize frame for faster processing
    process_frame = resize_if_needed(frame)

    current_time = time.time()
    img = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Run detection with error handling
        results = model(img)
        detections = results.xyxy[0]
    except Exception as e:
        print(f"Detection error: {e}")
        return display_frame  # Return original frame on error

    # Track objects in current frame
    current_objects = {}
    new_detections = []

    # Calculate ratio between original and processed frame if resized
    h_ratio = frame.shape[0] / process_frame.shape[0]
    w_ratio = frame.shape[1] / process_frame.shape[1]
    
    # Always display the current distance measurement on the frame
    if measured_distance > 0:
        distance_text = f"Distance: {measured_distance:.2f}m"
        cv2.putText(display_frame, distance_text, 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)

    for *box, conf, cls in detections:
        confidence = float(conf)

        # Skip if confidence is below threshold
        if confidence < detection_threshold:
            continue

        # Scale coordinates back to original frame size if we resized
        box = [float(coord) for coord in box]
        if process_frame.shape != frame.shape:
            x1, y1, x2, y2 = box
            x1 *= w_ratio
            x2 *= w_ratio
            y1 *= h_ratio
            y2 *= h_ratio
            box = [x1, y1, x2, y2]
            
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]

        # Use actual distance if available, otherwise estimate
        estimated_distance = measured_distance
        if estimated_distance <= 0:
            # Fall back to visual estimation if sensor reading unavailable
            box_height = y2 - y1
            frame_height = frame.shape[0]
            relative_size = box_height / frame_height
            # Rough distance estimate (would need calibration in real use)
            estimated_distance = round(1 / (relative_size + 0.1), 1)

        distance_text = f"{estimated_distance:.2f}m"

        # Determine direction based on x position
        x_pos = (x1 + x2) // 2
        frame_width = frame.shape[1]

        if x_pos < frame_width * 0.33:
            direction = "left"
        elif x_pos > frame_width * 0.66:
            direction = "right"
        else:
            direction = "ahead"

        # Store object with its position
        current_objects[label] = {
            'confidence': confidence,
            'position': (x_pos, (y1 + y2) // 2),
            'distance': estimated_distance,
            'direction': direction
        }

        # Add to detections list for WebSocket
        new_detections.append({
            'type': label,
            'confidence': float(confidence),
            'distance': float(estimated_distance),
            'direction': direction,
            'position': {'x': int(x_pos), 'y': int((y1 + y2) // 2)}
        })

        # Draw bounding box - use display_frame since that's what we return
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{label} {confidence:.2f} {distance_text}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Update current detections for WebSocket
    current_detections = new_detections

    # Speak about objects with cooldown
    for label, info in current_objects.items():
        # Check if we should announce this object
        if label not in last_spoken or (current_time - last_spoken[label]) > speak_cooldown:
            direction = info['direction']

            # Announce object with direction and distance
            speak_text = f"{label} {direction}, {info['distance']:.1f} meters"
            speak(speak_text)

            # Update last spoken time
            last_spoken[label] = current_time

    # Clean up old objects from last_spoken
    for label in list(last_spoken.keys()):
        if label not in current_objects and (current_time - last_spoken[label]) > 10:
            del last_spoken[label]

    return display_frame
