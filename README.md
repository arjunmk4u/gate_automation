# Automatic Gate Authentication System

This project is a prototype for an automated gate system using Computer Vision (YOLOv8 + EasyOCR) and Python (Gradio).

## Features
- **Real-time Vehicle Detection**: Uses YOLOv8 to detect vehicles.
- **License Plate Recognition**: Uses EasyOCR to extract and validate Indian number plates.
- **Authentication**: Checks recognized plates against a SQLite database.
- **Guard Panel**: 
    - Live camera feed with bounding boxes and status overlays.
    - **Gate Animation**: Visual line indicating gate status (Red Horizontal = Closed, Green Vertical = Open).
    - **Manual Approval**: "Accept" and "Decline" buttons for unauthorized vehicles.
- **Admin Panel**: Manage authorized vehicles and view access logs.

## Installation

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Application**:
    ```bash
    python app.py
    ```
2.  **Open the Browser**:
    -   The app usually runs at `http://127.0.0.1:7860`.
3.  **Admin Panel**:
    -   Add authorized plates (e.g., `KA01AB1234`).
4.  **Guard Panel**:
    -   The camera feed starts automatically.
    -   **Authorized Plate**: Gate opens automatically.
    -   **Unauthorized Plate**: Alert appears. Guard must click "✅ ALLOW ACCESS" or "⛔ DENY ACCESS".

## Project Structure
- `app.py`: Main Gradio application.
- `detection.py`: YOLO and OCR logic.
- `database.py`: SQLite database operations.
- `requirements.txt`: Python dependencies.

## Notes
- To reset the gate from "Open" to "Closed", the system uses a 5-second timer or manual detection changes.
- Ensure only one instance is running to avoid camera conflicts.
