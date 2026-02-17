import gradio as gr
import cv2
import pandas as pd
import time
import os
import database
import numpy as np
import threading
import multiprocessing
import queue
from datetime import datetime

# Initialize Database
database.initialize_db()

# --- Multiprocessing Detection Logic ---
def run_detection_process(input_q, output_q):
    """
    Worker process function for license plate detection.
    Loads the model independently to bypass GIL.
    """
    try:
        from detection import GateDetector
        detector = GateDetector()
        print("✅ Detection Process: Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Detection Process Setup Error: {e}")
        return

    while True:
        try:
            frame = input_q.get()
            
            # Detect
            detections = detector.detect_and_recognize(frame)
            
            # Clear old results to keep it real-time
            while not output_q.empty():
                try:
                    output_q.get_nowait()
                except queue.Empty:
                    pass
            
            output_q.put(detections)
            
        except Exception as e:
            print(f"Detection Process Loop Error: {e}")
            time.sleep(0.1)

# Global queues and process
input_queue = None
output_queue = None
detection_process = None

# Global State
class SystemState:
    def __init__(self):
        self.gate_status = "Closed" # "Closed", "Open"
        self.gate_open_time = 0
        self.current_plate = None
        self.last_detection_time = 0
        self.approval_status = "Idle" # "Idle", "Pending", "Approved", "Denied"
        self.pending_plate = None
        self.current_detections = [] 

state = SystemState()

# --- Helper Functions ---
def get_logs_df():
    logs = database.get_logs()
    if not logs:
        return pd.DataFrame(columns=["id", "plate_number", "timestamp", "action", "image_path"])
    return pd.DataFrame(logs)

def get_plates_df():
    plates = database.get_all_plates()
    if not plates:
        return pd.DataFrame(columns=["id", "plate_number", "owner_name", "created_at"])
    return pd.DataFrame(plates)

def add_plate(plate, owner):
    if not plate or not owner:
        return "Error: Please fill all fields.", get_plates_df()
    success, msg = database.add_authorized_plate(plate, owner)
    return msg, get_plates_df()

def remove_plate(plate):
    if not plate:
        return "Error: select a plate", get_plates_df()
    success, msg = database.remove_authorized_plate(plate)
    return msg, get_plates_df()

# --- Camera Handling ---
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Try to set higher resolution for long-range detection
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

# Global camera instance
global_camera = None

def get_current_frame():
    """
    Fetches the latest frame, processes it, and returns it.
    This is called periodically by the Timer.
    """
    global input_queue, output_queue, global_camera
    
    # Initialize camera if not already
    if global_camera is None:
         try:
            global_camera = Camera()
            print("Camera started.")
         except:
             return np.zeros((720, 1280, 3), dtype=np.uint8)

    frame = global_camera.get_frame()
    if frame is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    # Remove resize to keep full resolution for detection
    # frame = cv2.resize(frame, (640, 480))
    
    # --- Multiprocessing Integration ---
    if input_queue is not None:
        # 1. Update Detections from worker process
        try:
            while not output_queue.empty():
                state.current_detections = output_queue.get_nowait()
        except queue.Empty:
            pass
        
        # 2. Push Frame to worker process (Non-blocking)
        try:
            if input_queue.empty():
                input_queue.put_nowait(frame)
        except:
            pass
    
    # --- Logic ---
    try:
        # 1. Gate Auto-Close Logic
        if state.gate_status == "Open":
            if time.time() - state.gate_open_time > 5: # Close after 5 seconds
                state.gate_status = "Closed"
                state.current_plate = None
                state.approval_status = "Idle"

        # 2. Verify Approval (Manual)
        if state.approval_status == "Approved":
            state.gate_status = "Open"
            state.gate_open_time = time.time()
            database.log_access(state.pending_plate, "Manual Approval")
            state.approval_status = "Idle" 
            state.pending_plate = None
        elif state.approval_status == "Denied":
            database.log_access(state.pending_plate, "Manual Denial")
            state.approval_status = "Idle" 
            state.pending_plate = None

        # 3. Detection Processing (Using latest results from process)
        if state.gate_status == "Closed":
            detections = state.current_detections # Get last known detections
            
            for d in detections:
                plate_text = d['plate']
                bbox = d['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Debounce detection
                is_new_plate = state.current_plate != plate_text
                # If it's a new plate, process immediately.
                # If it's the *same* plate, process again after 2 seconds (faster retry for better OCR readings)
                if is_new_plate or (time.time() - state.last_detection_time > 2.0):
                    
                    is_authorized = database.check_access(plate_text)
                    
                    if is_authorized:
                        state.gate_status = "Open"
                        state.gate_open_time = time.time()
                        state.approval_status = "Idle"
                        database.log_access(plate_text, "Authorized Access")
                        cv2.putText(frame, "ACCESS GRANTED", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        state.approval_status = "Pending"
                        state.pending_plate = plate_text
                        cv2.putText(frame, "UNAUTHORIZED", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    state.current_plate = plate_text
                    state.last_detection_time = time.time()

        # 4. Draw Status Overlays
        if state.approval_status == "Pending":
            cv2.putText(frame, f"UNAUTHORIZED: {state.pending_plate}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "WAITING FOR ACTION...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        if state.gate_status == "Closed":
            cv2.line(frame, (center_x - 100, center_y), (center_x + 100, center_y), (0, 0, 255), 10)
            cv2.putText(frame, "GATE CLOSED", (center_x - 60, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.line(frame, (center_x - 120, center_y - 50), (center_x - 120, center_y + 50), (0, 255, 0), 10)
            cv2.line(frame, (center_x + 120, center_y - 50), (center_x + 120, center_y + 50), (0, 255, 0), 10)
            cv2.putText(frame, "GATE OPEN", (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error in processing loop: {e}")
        return frame

def approve_action():
    if state.approval_status == "Pending":
        state.approval_status = "Approved"
        return "Approved access."
    return "No pending request."

def deny_action():
    if state.approval_status == "Pending":
        state.approval_status = "Denied"
        return "Denied access."
    return "No pending request."

# --- App Layout ---
with gr.Blocks(title="Gate Automation System") as demo:
    gr.Markdown("# 🚧 Automatic Gate Control System")
    
    with gr.Tabs():
        with gr.Tab("👮 Guard Panel"):
            with gr.Row():
                with gr.Column(scale=2):
                    live_feed = gr.Image(label="Live Camera Feed", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Manual Controls")
                    status_text = gr.Textbox(label="Status", value="System Ready", interactive=False)
                    with gr.Row():
                        accept_btn = gr.Button("✅ ALLOW ACCESS", variant="primary")
                        deny_btn = gr.Button("⛔ DENY ACCESS", variant="stop")
                    
                    gr.Markdown("---")
                    gr.Markdown("**Instructions:**\n- If an unauthorized plate is detected, the 'PENDING' alert will appear.\n- Use the buttons above to control the gate manually.")

            # Timer for live feed (20 FPS)
            timer = gr.Timer(0.05)
            timer.tick(get_current_frame, outputs=live_feed)
            
            # Use every=0.05 for polling
            # demo.load(get_current_frame, inputs=None, outputs=live_feed, every=0.05)
            
            accept_btn.click(approve_action, inputs=None, outputs=status_text)
            deny_btn.click(deny_action, inputs=None, outputs=status_text)

        with gr.Tab("🔧 Admin Panel"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Add Authorized Plate")
                    plate_input = gr.Textbox(label="Plate Number", placeholder="KA01AB1234")
                    owner_input = gr.Textbox(label="Owner Name", placeholder="John Doe")
                    add_btn = gr.Button("Add Plate")
                    result_msg = gr.Markdown()
                
                with gr.Column():
                    gr.Markdown("### Remove Plate")
                    remove_input = gr.Textbox(label="Plate Number to Remove")
                    remove_btn = gr.Button("Remove Plate")
            
            gr.Markdown("### Authorized Vehicles")
            plates_table = gr.Dataframe(value=get_plates_df, label="Registered Plates", interactive=False, headers=["ID", "Plate", "Owner", "Added At"])
            
            gr.Markdown("### Access Logs")
            logs_table = gr.Dataframe(value=get_logs_df, label="Access History", interactive=False, headers=["ID", "Plate", "Time", "Action", "Image"])
            
            refresh_btn = gr.Button("Refresh Data")
            
            add_btn.click(add_plate, inputs=[plate_input, owner_input], outputs=[result_msg, plates_table])
            remove_btn.click(remove_plate, inputs=[remove_input], outputs=[result_msg, plates_table])
            refresh_btn.click(lambda: (get_plates_df(), get_logs_df()), inputs=None, outputs=[plates_table, logs_table])

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    
    # Initialize Queues
    input_queue = multiprocessing.Queue(maxsize=1)
    output_queue = multiprocessing.Queue(maxsize=1)
    
    # Start Detection Process
    detection_process = multiprocessing.Process(
        target=run_detection_process, 
        args=(input_queue, output_queue),
        daemon=True
    )
    detection_process.start()
    
    print("Process started. Launching UI...")
    
    # Launch Gradio
    demo.launch()
