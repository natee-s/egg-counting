# PROJECT SPECIFICATION: AI Egg Counting & Defect Detection System

## 1. Project Overview
A real-time Computer Vision system designed to detect, track, count, and classify eggs on a moving roller conveyor belt. The system prioritizes **100% counting accuracy** and runs efficiently on a **CPU-only environment (Intel Graphics)** using a decoupled architecture (Python Backend + Web Frontend) with a Sci-Fi/Gamification UI.

## 2. Hardware Constraints & Mitigation Strategy
*   **Target Hardware:** CPU only (Intel Graphics 128 MB VRAM). No dedicated GPU.
*   **Mitigation 1 (Inference Engine):** All models MUST be exported to **Intel OpenVINO** format (`.xml`, `.bin`) to accelerate CPU inference.
*   **Mitigation 2 (Model Size):** Use Nano versions of YOLO models (e.g., YOLOv8n-seg, YOLOv8n-cls).
*   **Mitigation 3 (Smart Pipeline):** Do not run defect classification on every frame. Only run the classification model on a cropped bounding box of an egg *exactly when* its centroid crosses the virtual counting line.

## 3. System Architecture (Decoupled)
The system is divided into two main components communicating via WebSocket.

### 3.1 Backend (Python)
*   **Core Frameworks:** FastAPI (for API/WebSocket), OpenCV, Ultralytics (YOLO), OpenVINO toolkit.
*   **CV Pipeline Logic:**
    1.  **Frame Extraction:** Read video frame.
    2.  **Instance Segmentation:** YOLOv8n-seg (OpenVINO) detects eggs and outputs segmentation masks + bounding boxes.
    3.  **Tracking:** `ByteTrack` algorithm assigns a unique ID to each egg to handle occlusion and rolling.
    4.  **Counting (Line Crossing):** Define a Virtual Line. Monitor the `(x, y)` centroid of each tracked ID. When the centroid crosses the line, increment the `Total Count`.
    5.  **Defect Detection (Event-Triggered):** Upon crossing the line, crop the egg using its bounding box, pass it to YOLOv8n-cls (Classification model: Normal vs. Cracked/Dirty). Increment respective `Normal Count` or `Defect Count`.
    6.  **Streaming:** Compress the frame with drawn graphics to base64 or MJPEG. Send frame and JSON payload via WebSocket.

### 3.2 Frontend (Web App)
*   **Core Frameworks:** React.js, Socket.IO-client / native WebSocket, Framer Motion (for animations), Tailwind CSS.
*   **UI/UX Concept:** Sci-Fi / Cyberpunk / Gamification.
*   **Role:** Receive WebSocket stream, render the video feed on a Canvas, and overlay gamified UI elements and animations based on the JSON payload.

## 4. Communication Protocol (WebSocket Payload)
The Backend will emit a JSON payload per frame to the Frontend.

**Event Name:** `frame_update`
**Payload Structure:**
```json
{
  "frame_data": "data:image/jpeg;base64,...", // Encoded image with bounding boxes & masks drawn
  "stats": {
    "fps": 15.2,
    "total_count": 142,
    "normal_count": 138,
    "defect_count": 4
  },
  "events": [
    {
      "id": 15,
      "type": "count_triggered",
      "status": "normal", // or "defect"
      "x": 320, // Coordinate on line crossing for UI particle effects
      "y": 240
    }
  ]
}
```

## 5. UI/UX & Animation Specifications (Gamification)
The Frontend MUST implement the following visual elements:

*   **Theme:** Dark mode, Neon accents (Cyan/Blue for normal, Neon Red/Orange for defects), Monospaced fonts (e.g., Roboto Mono).
*   **Video Feed Overlay:**
    *   **The Virtual Line:** A glowing neon laser line across the conveyor.
*   **HUD (Heads-Up Display) Elements:**
    *   **Combo/Total Score:** Large glowing text at the top center. When `total_count` increases, the number scales up and down briefly (Pop-up effect).
    *   **Status Bars/Gauges:** Circular progress bars or Sci-Fi meters showing the ratio of Normal vs. Defect eggs.
*   **Event Animations (Crucial for Gamification):**
    *   Whenever an `events` array contains a new crossing event:
        *   **Hit Effect:** Spawn a brief particle burst or a "+1" floating text at the `(x, y)` coordinates of the crossing.
        *   **Color Coding:** If `status == "normal"`, emit Cyan particles. If `status == "defect"`, emit Red particles with a warning glitch effect or screen shake on the UI card.

## 6. Implementation Phases (For AI Code Generation)

*   **Phase 1: Base CV Pipeline (Python)**
    *   Setup OpenCV to read the video.
    *   Integrate YOLOv8n-seg for detection.
    *   Integrate ByteTrack for tracking IDs.
*   **Phase 2: Counting & Optimization**
    *   Implement Line Crossing logic.
    *   Convert models to OpenVINO format.
*   **Phase 3: Defect Detection Trigger**
    *   Implement cropping logic on line cross.
    *   Run dummy classification (or real YOLOv8n-cls) on the crop.
*   **Phase 4: Backend WebSocket Server**
    *   Wrap the loop in a FastAPI application.
    *   Emit the base64 frames and stats JSON.
*   **Phase 5: React Frontend & Gamification UI**
    *   Build the React dashboard.
    *   Connect WebSocket.
    *   Implement Framer Motion animations for the "+1" pop-ups, glowing counters, and particle effects based on the `events` payload.