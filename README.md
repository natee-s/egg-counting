# OVO · Egg Counting & Defect Detection

ระบบตาม [project_spec.md](project_spec.md): Python/FastAPI + OpenCV + YOLO
detection/segmentation + ByteTrack + OpenVINO CPU และ React video viewer
สื่อสารด้วย **native WebSocket** ไม่ใช่ Socket.IO

โปรเจกต์ตั้งค่าเป็น **real mode** ใช้โมเดล `Kaggle Egg Detection YOLO11n`
ที่ผู้ใช้เลือก พร้อม OpenVINO CPU + ByteTrack และเส้นเฉียงตามแนวสีส้มในภาพอ้างอิง.
ระบบเริ่มหยุดอยู่จนกด Play และนับจากเฟรมวิดีโอจริงโดยไม่มี events จำลอง.
โมเดลนี้ตรวจจับ class egg ด้วยกรอบสี่เหลี่ยม จึงปิดการตรวจรอยร้าว/สิ่งสกปรกไว้.

หน้าจอเหลือวิดีโอกับกล่องนับมุมขวาบนตามคำขอผู้ใช้. กรอบสีน้ำเงิน = ยังไม่นับ,
สีเขียว = นับแล้ว (ไม่ใช่ผลตรวจคุณภาพ). จุดแดงคือศูนย์กลางที่ใช้ตัดสินการข้ามเส้น.
Eggs = ยอดข้ามเส้นสะสม, Detections = จำนวนกรอบที่โมเดลพบในเฟรม,
Tracked = จำนวนที่ ByteTrack ยืนยัน. ไม่มี HUD, particles, charts หรือ sidebar.
ผลเปรียบเทียบค่า tracker ล่าสุดอยู่ใน [KAGGLE_RUNTIME.md](KAGGLE_RUNTIME.md).

## วิดีโอทดสอบที่ตั้งค่าไว้

ไฟล์ `Eggs On A Conveyor Stock Video.mp4` ใน project root อ่านได้: **854 × 480**,
**25 FPS**, **267 เฟรม**, ความยาว **10.68 วินาที**. `backend/.env` ตั้งไว้:

```dotenv
EGG_MODE=real
EGG_SOURCE="Eggs On A Conveyor Stock Video.mp4"
EGG_SEG_MODEL=models/candidates/kaggle_egg_yolo11n_openvino_model
EGG_IMAGE_SIZE=640
EGG_EGG_LABELS=egg
EGG_DEFECT_ENABLED=0
EGG_START_PAUSED=1
EGG_LINE_POINTS=0.337:0,0.40:0.12,0.425:0.20,0.50:0.31,0.66:0.51,0.85:0.72,0.975:0.885
```

รัน `./start-video.ps1` หรือเปิด Backend/Frontend ด้วยคำสั่งด้านล่าง.
หน้าเว็บเปิดเฟรมแรกและเริ่ม **paused**. กด Play ในกล่องมุมขวาบนเพื่อประมวลผลและนับ.
Pause หยุด worker และการอ่านเฟรมจริง โดยรอเฟรมที่กำลังประมวลผลให้จบก่อนตอบรับ.
ประมวลผลทุกเฟรมตามลำดับ ความเร็วเล่นขึ้นกับ CPU. ไม่เล่นอัตโนมัติและไม่วนซ้ำเอง.
เมื่อจบ กด Replay เพื่อเริ่ม session ใหม่. Reset ยังใช้ได้ผ่าน API เพื่อกลับเฟรมแรกแบบหยุดอยู่.
ไม่รองรับ seek ในโหมดนับ เพราะการข้ามหรือย้อนเฟรมทำให้สถานะ tracking และยอดนับปะปน.
หากต้องการดู MP4 อย่างเดียวพร้อม seek ให้เลือก `EGG_MODE=video`.

โมเดลที่เปิดใช้ผ่านการเลือกโดยผู้ใช้; ได้ track 72/72 จุดอ้างอิงในภาพบางส่วน
จึงยังไม่ใช่การยืนยันยอดนับถูกต้อง 100%. ดู [ผลทดสอบ](KAGGLE_RUNTIME.md).
เส้นบนหน้าจอและใน state machine ใช้ polyline เดียวกันเป็นช่วงเส้นจำกัดความยาว.
นับเฉพาะไข่ที่ติดตามจากด้านต้นทางแล้วตัดผ่านเส้นไปด้านปลายทาง ไม่รวมการผ่านเลยปลายเส้น.
ไข่ที่นับแล้วไม่ re-arm เมื่อกลิ้งกลับ. การกู้ ID ใช้ motion/size และยังมีข้อจำกัดจากการบังกัน.
ผล crossing ใช้ `status=unknown`, `label=not_inspected`, `confidence=null`;
HUD แสดง COUNTED และ Not inspected โดยไม่ตีความสีเปลือกเป็นคุณภาพของไข่.
ไฟล์รายงาน 267 เฟรมใน metadata แต่ OpenCV ถอดได้ 262 เฟรม; บันทึกจำนวนที่ถอดได้เมื่อจบ.

## โครงสร้าง

```text
14_Egg_counting/
├── project_spec.md
├── README.md
├── start-demo.ps1
├── start-video.ps1             # start AI counting of real MP4, initially paused
├── backend/
│   ├── main.py                 # CV, state machine, persistence, FastAPI/WS
│   ├── export_models.py        # .pt → OpenVINO XML/BIN/metadata
│   ├── .env.example
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── tests/test_pipeline.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx             # live HUD, Canvas, events, reconnect, CSV
│       ├── styles.css          # responsive neon HUD + CSS particles
│       └── main.jsx
├── models/                     # custom-trained OpenVINO models
└── data/                       # SQLite event ledger + inference cache
```

## ติดตั้ง (PowerShell, Windows)

ใช้ **Python 3.12 x64**, Node.js 20.19+ หรือ 22.12+ และ npm รุ่นที่รองรับ
เครื่องนี้มี Python 3.14 เป็นค่าเริ่มต้น ซึ่งไม่ใช่ environment ที่ทดสอบสำหรับ
ชุด CV ที่ pin ไว้ จึงสร้าง `.venv` ด้วย Python 3.12 และใช้ Python จาก `.venv`
โดยตรง ไม่จำเป็นต้องเปิด execution policy เพื่อ activate environment

```powershell
cd D:\12.Project\14_Egg_counting
py -3.12 -m venv .venv

# ติดตั้ง PyTorch CPU ก่อน เพื่อไม่ดึง CUDA packages
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
Copy-Item backend/.env.example backend/.env

cd frontend
npm ci
```

Demo ใช้แค่ FastAPI, uvicorn, websockets, OpenCV, NumPy และ python-dotenv;
Ultralytics/OpenVINO ถูก import เฉพาะ real mode. Dependencies ติดตั้งไว้แล้ว
ใน `.venv` และ `frontend/node_modules` สำหรับ workspace นี้

## รัน

เปิด Terminal 1 จาก project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

เปิด Terminal 2:

```powershell
cd frontend
npm run dev
```

เปิด **http://127.0.0.1:5173**. API docs: **http://127.0.0.1:8000/docs**.
หรือรัน `./start-video.ps1` เพื่อเริ่มทั้งสอง service ใน background แล้วกด Enter
เพื่อหยุด service ที่ script เปิดเอง. Log อยู่ใน `data/`.
`./start-demo.ps1` เป็นคำสั่งเลือกใช้ภาพจำลองโดยเจตนาเท่านั้น.

อย่าใช้หลาย worker: กล้อง, ByteTrack และ state ของการนับต้องมีเจ้าของเพียง process
เดียว. อย่าเปิด `--reload` บนสายพานที่กำลังใช้งานจริง เพราะการ reload สร้าง session ใหม่.

ใน video mode ปุ่ม Play/Pause หยุดหรือเล่น **ไฟล์ MP4 จริง**.
ใน real mode Play/Pause หยุด **การประมวลผลและการนับจริง**.
ใน demo mode Pause หยุด **การแสดงภาพ** เท่านั้น.
กรอบ/จุดศูนย์กลาง/เส้นวาดจาก Backend. หน้าเว็บไม่มี Toggle HUD หรือ Export;
event ledger ของแต่ละ session ยังอ่านได้ผ่าน `/api/events`.

## โมเดลจริง / OpenVINO

ผลวิเคราะห์แยก detection → tracking → crossing และการเทียบโมเดล Kaggle
YOLO11n บน CPU อยู่ใน [KAGGLE_COMPARISON.md](KAGGLE_COMPARISON.md).
โมเดล Kaggle เปิดใช้ตามคำขอแล้ว; รายงานดังกล่าวเป็น baseline ก่อนปรับ ByteTrack.

เปิดใช้โมเดล Kaggle YOLO11n ที่ผู้ใช้เลือกจาก `models/candidates/` สำหรับนับไข่แล้ว.
ผลตรวจเบื้องต้นยังไม่ยืนยันความแม่นยำการนับ 100%; ส่วนตรวจตำหนิยังปิดไว้.
ตัวเลือกและข้อจำกัดของ license อยู่ใน [ผลทดสอบ](MODEL_EVALUATION.md).
หากต้องการปรับโมเดลให้เหมาะกับสายพานจริงและเพิ่มการตรวจตำหนิ:

1. **YOLOv8n-seg** ด้วย instance masks ของไข่บนสายพาน รวมภาพไข่ชิดกัน,
   บังกัน, กลิ้งย้อน, แสงสะท้อน และ label `egg`.
2. **YOLOv8n-cls** ด้วย crops จากมุมกล้องเดียวกัน: `normal` + `defect`
   หรือ `normal`, `cracked`, `dirty`. แบ่ง train/val/test ตามวันหรือคลิป
   เพื่อไม่ให้ภาพเฟรมข้างเคียงรั่วไปอยู่ทั้ง train และ test.

Stock COCO weights ไม่ใช่โมเดลตรวจจับไข่; ImageNet weights ไม่ใช่โมเดลตรวจตำหนิ.
ระบบ real mode จะตรวจ class labels และรายงาน error หากโมเดลไม่ถูกต้อง.

Export จาก custom checkpoints:

```powershell
.\.venv\Scripts\python.exe backend/export_models.py --seg models/egg_seg.pt --cls models/egg_cls.pt
```

Export แบบ fixed shapes, batch 1, FP32: segmentation **416** และ classification
**224**. ตั้ง `.env` ให้มี image sizes เดียวกับตอน export. สร้าง folders
`models/egg_seg_openvino_model` และ `models/egg_cls_openvino_model` ซึ่งแต่ละ folder
มี `.xml`, `.bin`, `metadata.yaml`. ไม่มีการ export/download weights ทุกครั้งที่เริ่ม.
ถ้า output folder มีอยู่แล้ว tool จะไม่เขียนทับ ให้เลือก `--seg-out` / `--cls-out` ใหม่.

ใน `backend/.env`:

```dotenv
EGG_MODE=real
EGG_SOURCE=0
EGG_SEG_MODEL=models/egg_seg_openvino_model
EGG_CLS_MODEL=models/egg_cls_openvino_model
EGG_EGG_LABELS=egg
EGG_DEFECT_ENABLED=1
EGG_LINE_POINTS=
EGG_LINE_AXIS=y
EGG_LINE_POSITION=0.55
EGG_DIRECTION=1
```

`EGG_SOURCE` รับ camera index, local video เช่น `data/conveyor.mp4`, หรือ URL
ของกล้อง. Video files อ่านทุกเฟรมและ **ไม่วนซ้ำ**; เมื่อจบจะส่ง `completed`.
Camera disconnect จะส่ง error และหยุดแทนการเชื่อมใหม่โดยสูญเสีย identity แบบเงียบ ๆ.

Runtime ใช้ `intel:cpu` และ recompile ด้วย device `CPU`, `LATENCY`, หนึ่ง stream,
จำกัด inference threads (`EGG_CPU_THREADS=2`). GPU/VRAM ไม่ถูกใช้ในการ inference.
Ultralytics preprocessing/postprocessing และ ByteTrack ใช้ CPU เช่นกัน.
Frontend ใช้ CSS effects ที่จำกัดจำนวน พร้อมรองรับ prefers-reduced-motion.

Optional INT8 ต้องมี representative calibration data และ `nncf`; เปรียบเทียบ
ความแม่นยำกับ FP32 ก่อนใช้งานจริง ห้ามใช้ COCO/ImageNet calibration แทนข้อมูลไข่:

```powershell
.\.venv\Scripts\python.exe -m pip install nncf
.\.venv\Scripts\python.exe backend/export_models.py --seg models/egg_seg.pt --cls models/egg_cls.pt --int8 --seg-data data/eggs.yaml --cls-data data/egg_quality --seg-out models/egg_seg_int8_openvino_model --cls-out models/egg_cls_int8_openvino_model
```

## การนับและขอบเขตความแม่นยำ

State machine: `UNARMED → ARMED → COUNTED`.

- ไข่ต้องถูกพบ upstream ห่างจากเส้นอย่างน้อย `EGG_HYSTERESIS_PX` ติดต่อกัน
  `EGG_ARM_FRAMES` ครั้ง จึง arm. ไข่ที่เริ่มถูกพบ downstream จะไม่ถูกนับทันที.
- เมื่อ signed centroid distance เปลี่ยนจาก `< 0` เป็น `>= 0` ในทิศที่ตั้งไว้
  จะเปลี่ยนเป็น `COUNTED` ที่ไม่ย้อนกลับ จึงไม่นับซ้ำจากการกลิ้งย้อนสำหรับ identity เดิม.
- Segmentation centroid ใช้ moments ของ instance polygon; fallback เป็น box center
  ถ้า mask ไม่มีพื้นที่. Crossing coordinates คำนวณจุดตัดระหว่างสอง observations.
- ByteTrack ใช้ high-confidence association และ low-confidence rescue พร้อม lost
  buffer. Tracker อัปเดตแม้เฟรมที่ไม่พบไข่. Detection confidence 0.10 ต้องต่ำพอ
  ให้ low-confidence rescue ทำงาน; ไม่กรองทิ้งด้วย threshold สูงก่อนเข้า tracker.
- เมื่อ ID เปลี่ยนในช่วงสั้น ใช้ predicted motion + box size และโอน physical
  identity/สถานะเฉพาะเมื่อ match เป็น **หนึ่งต่อหนึ่งและมี candidate เดียว**.
  ไม่รวมไข่สองฟองด้วย cooldown หรือการห้ามนับทั้งบริเวณเส้น.
- Ambiguous match จะสร้าง unarmed identity และเพิ่ม `ambiguous_ids` ให้ operator
  ตรวจสอบ. Gap ยาวจะ reset arming และเพิ่ม `long_gaps`. วิธีนี้อาจพลาดการนับ
  ในกรณีกำกวมและไม่ใช่การพิสูจน์ physical identity.
- Defect classification เรียกครั้งเดียวจาก **raw crop ของเฟรมแรกที่พบว่าข้ามเส้น**
  ก่อนวาด overlay; มี blank-image warmup หนึ่งครั้งต่อโมเดลตอน startup.
  เวลา crossing จริงอยู่ระหว่างสองเฟรม จึงไม่ใช่การ sampling ระดับต่อเนื่อง.
- ตรวจผิดพลาดจะนับเป็น `unknown` และเพิ่ม `unclassified_count` โดยไม่แอบนับเป็น good.
- SQLite `UNIQUE(run_id, physical_id)` ป้องกัน event ซ้ำใน session และ commit
  ก่อน broadcast. ทุกการเริ่ม backend สร้าง **session ใหม่**; ประวัติเดิมยังอยู่
  แต่ไม่ resume physical IDs จากก่อน restart. ควรหยุดสายพานและเคลียร์บริเวณเส้น
  ก่อนเริ่ม session ใหม่เพื่อหลีกเลี่ยงการนับไข่ที่ค้างอยู่ซ้ำ.

**ไม่สามารถรับรอง 100% counting accuracy จากโค้ดอย่างเดียว** เมื่อไข่บังกันจน
มองไม่เห็น, tracker สลับ identity, กล้อง/CPU พลาดเฟรม หรือไข่เกิดใหม่ใกล้เส้น.
การตรวจ cracks/dirt แบบ snapshot ก็ตรวจได้เฉพาะพื้นผิวที่มองเห็นขณะข้ามเส้น.
ต้องจัดแสง/มุมกล้อง/ความเร็วสายพานให้เห็นไข่แยก instance ได้ และอาจต้องเพิ่ม
กล้องหรือ encoder/sensor ถ้าต้องรับประกันการนับเชิงกายภาพทุกฟอง.

ไม่มี hard-coded FPS claim: เปรียบเทียบ `fps` กับ `source_fps` บนเครื่องจริง.
File processing ไม่ข้ามเฟรม แต่ throughput เป็นเวลาประมวลผล ไม่ใช่ playback speed.
`eggs_per_minute` คือยอดในช่วง 60 วินาที **wall clock**; ใน video file mode
ไม่ใช่ค่าอัตราบนสายพานจริง. Event `media_seconds` ใช้ source FPS สำหรับ video
frame indexing; กล้องที่รายงาน FPS ผิดต้องเทียบ timestamp เพิ่มเติม.
หาก CPU ไม่ทันกล้อง driver อาจทิ้งเฟรมได้แม้แอปไม่ได้ตั้งให้ skip frame.

## Protocol

`GET /api/health`, `GET /api/stats`, `GET /api/events?after=0&limit=500&run_id=...`.

WebSocket: `/ws?run_id=<known-run>&after=<last-received-event-sequence>`.
รูปแบบตามสเปกพร้อม metadata สำหรับ session และ replay:

```json
{
  "type": "frame_update",
  "event": "frame_update",
  "run_id": "session-id",
  "mode": "real",
  "state": "running",
  "sequence": 15,
  "timestamp": "2026-10-05T06:00:00+00:00",
  "frame_data": "data:image/jpeg;base64,...",
  "frame_size": {"width": 960, "height": 540},
  "line": {"axis": "y", "position": 0.55, "direction": 1},
  "stats": {
    "fps": 15.2, "total_count": 15, "normal_count": 14,
    "defect_count": 1, "unclassified_count": 0, "eggs_per_minute": 15
  },
  "events": [{
    "sequence": 15, "run_id": "session-id", "id": 15, "tracker_id": 23,
    "type": "count_triggered", "status": "good", "legacy_status": "normal",
    "total_count": 15, "fps": 15.2, "timestamp": "2026-10-05T06:00:00+00:00",
    "label": "normal", "confidence": 0.98, "x": 320, "y": 297
  }]
}
```

`status` ใช้ `good|defect` ตาม user requirement; `legacy_status=normal` รองรับ
คำเรียกในสเปก. `unknown` เป็น extension เมื่อ classifier ล้มเหลว.
`snapshot` ส่ง stats ปัจจุบัน, `event_replay` ส่ง events ที่ขาด, `heartbeat`
แจ้งสถานะเมื่อไม่มี frame ใหม่. Frontend ใช้ authoritative stats และไม่บวกยอดจาก
events ซ้ำ. Replay ไม่มี particle เก่าย้อนหลัง. Client ช้าจน queue เต็มจะถูกปิด
ด้วย code 1013 แล้ว reconnect/replay จาก SQLite แทนการทำ CV worker ช้าลง.

Vite dev server proxy `/ws` และ `/api`. หาก host frontend แยก ต้องตั้ง
`VITE_WS_URL`, `VITE_API_URL` และ `EGG_ORIGINS` ให้ตรงกัน. Production build:

```powershell
cd frontend
npm run build
```

นำ `frontend/dist` ไปเสิร์ฟด้วย local reverse proxy ที่ forward `/api` และ
WebSocket `/ws` ไป backend port 8000. `vite preview` เสิร์ฟ static files อย่างเดียว
จึงต้องกำหนด URLs ของ backend ตอน build หรือใช้ reverse proxy.
ทั้งหมดรัน local หลังติดตั้ง dependencies และ export โมเดลแล้ว ไม่ใช้ cloud inference.

## ตรวจสอบ

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

Tests ครอบคลุมการข้ามเส้น, rollback, jitter, ไข่หลายฟอง, occlusion/ID recovery,
ambiguous recovery, ทิศทางเส้น, classifier trigger/failure, SQLite idempotency,
WebSocket replay, origin validation และ demo end-to-end. การทดสอบ OpenVINO
ใช้กราฟ synthetic เพื่อตรวจ runtime adapter; ไม่ใช่ trained model accuracy test.

Acceptance ก่อน production: ทำ ground-truth per-egg crossing labels บนคลิป
แยกจาก training แล้วเทียบ events ทั้งเวลา/ตำแหน่ง/identity เพื่อแยก **misses**
และ **duplicates** (ยอดรวมเท่ากันอาจมีทั้งพลาดและนับซ้ำ). ทดสอบ crowding,
reverse motion, pause, lighting changes, camera loss และ CPU load บนเครื่องเป้าหมาย.
ตำหนิต้องเทียบ precision/recall แยก crack/dirt พร้อม false negatives.

## References

- [Ultralytics OpenVINO integration](https://docs.ultralytics.com/integrations/openvino/)
- [Ultralytics v8.3.228 ByteTrack implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.228/ultralytics/trackers/byte_tracker.py)
- [Ultralytics v8.3.228 OpenVINO backend](https://github.com/ultralytics/ultralytics/blob/v8.3.228/ultralytics/nn/autobackend.py)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite)

ตรวจ license ของ Ultralytics (AGPL-3.0 / enterprise) ให้เหมาะกับการแจกจ่ายผลิตภัณฑ์.
