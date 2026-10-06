# ตรวจจุดพลาดและเปรียบเทียบโมเดลไข่ — 6 ตุลาคม 2026

## ผลที่ได้

รัน OpenVINO บน **CPU เท่านั้น** (Intel Core Ultra 5 225H, inference threads 2)
กับไฟล์ `Eggs On A Conveyor Stock Video.mp4` ทั้ง 262 เฟรมที่ OpenCV อ่านได้
ใช้เส้นสีส้มเดิม, ByteTrack และเงื่อนไขนับชุดเดียวกัน, confidence 0.10,
NMS IoU 0.65 และสูงสุด 300 detections/เฟรม โดยไม่ได้เปลี่ยน `backend/.env`
หรือโมเดลที่ Dashboard ใช้งานอยู่

| โมเดล / input | จุดอ้างอิงที่ detect | จุดที่ได้ track | detection ใกล้เส้นที่ได้ track¹ | crossing events² | ID กำกวม | median pipeline/เฟรม | pipeline FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| industoai YOLOv8n-seg / 416 (ใช้อยู่) | 71/72 | 41/72 | 1,704/4,713 | 204 | 0 | 185.4 ms | 5.65 |
| industoai YOLOv8n-seg / 640 | 69/72 | 67/72 | 3,269/4,853 | 263 | 2 | 348.7 ms | 2.90 |
| Kaggle YOLO11n-detect / 640 | 72/72 | 72/72 | 4,903/6,094 | 308 | 13 | 105.6 ms | 9.42 |

¹ จำนวน **การพบในแต่ละเฟรม** ภายในระยะ 35 px จากเส้น ไม่ใช่จำนวนไข่จริง
หรือจำนวนฟองที่หายไป ค่า track แสดง detection ที่ ByteTrack คืน observation
ในเฟรมนั้น ² เป็นผลจากอัลกอริทึม ไม่ใช่จำนวนไข่จริงที่ตรวจยืนยันแล้ว

ค่าความเร็วรวมการอ่านเฟรม, infer, ByteTrack, ตัวนับ และการเก็บ trace
ในเครื่องมือวิเคราะห์ แต่ไม่รวม JPEG/WebSocket/การวาดหน้าเว็บ จึงใช้เทียบ
สองโมเดลในงานนี้ ไม่ใช่ความเร็วปลายทางของ Dashboard คลิปต้นทาง 25 FPS:
ทั้งสามตัวเลือกยังประมวลผลทุกเฟรมไม่ทันอัตราคลิปบนเครื่องที่ทดสอบ

## แยกสาเหตุการพลาด

- โมเดลปัจจุบันที่ 416 detect จุดอ้างอิง 71/72 แต่ได้ track เพียง 41/72:
  ในชุดตัวอย่างนี้มี **30 จุดที่ detect แล้วไม่ได้ track**
- ใกล้เส้น มี 3,009 detection/เฟรมที่ไม่ถูก tracker คืน observation;
  2,319 รายการมี confidence ต่ำกว่า 0.25 ขณะที่ค่า ByteTrack ปัจจุบันคือ
  `track_high_thresh=0.25`, `track_low_thresh=0.10`, `new_track_thresh=0.30`.
  ส่วนที่ confidence สูงกว่านั้นอาจเกี่ยวกับการจับคู่ตัวตน/ความแออัด
  และต้องดู trace แต่ละเฟรมก่อนสรุป
- Trace ของโมเดลปัจจุบันพบ 26 การเปลี่ยนฝั่งเส้นในแถบ 35 px
  ก่อน track อยู่สถานะ `armed`; อีก 76 การเปลี่ยนฝั่งเกิดกับ track ที่นับแล้ว
  และถูกกฎป้องกันนับซ้ำกั้นไว้ ตัวเลขนี้เป็น **เหตุการณ์ในเฟรม**
  ไม่ใช่จำนวนไข่ที่พลาดหรือนับซ้ำ
- เพิ่ม input ของโมเดลเดิมเป็น 640 ทำให้จุดอ้างอิงที่ได้ track สูงขึ้น
  แต่ความเร็วลดเหลือ 2.90 FPS จึงไม่ใช่ทางแก้เดี่ยวที่เหมาะกับ CPU นี้
- Kaggle 640 ให้ track มากขึ้นและเร็วกว่า แต่มีการกู้ ID 56 ครั้ง
  และการกู้ ID กำกวม 13 ครั้ง เทียบกับโมเดลใช้อยู่ 16 และ 0 ครั้ง
  จึงยังไม่อาจตีความ 308 crossing events ว่าถูกกว่า 204

ภาพตัวอย่างเฟรม 120: `pipeline-diagnostics/current_reference_120.jpg`
และ `pipeline-diagnostics/kaggle_reference_120.jpg`.
สีเขียว = detect และ track, เหลือง = detect แต่ไม่ track, แดง = ไม่ detect.
รายละเอียดรายเฟรมและราย detection ดู JSON ในโฟลเดอร์เดียวกัน
(`current_diagnostics.json`, `current640_diagnostics.json`, `kaggle_diagnostics.json`).

## ข้อจำกัดของ Kaggle และการตัดสินใจ

[โน้ตบุ๊ก Kaggle](https://www.kaggle.com/code/uzairahmad1434/egg-detection-yolov11n)
ฝึก YOLO11n แบบ bounding-box detection จากภาพที่ชื่อเดียวกับคลิปนี้
57 ภาพ/5,617 instances และตั้ง `train: train/images` กับ
`val: train/images` เป็นภาพโฟลเดอร์เดียวกัน คะแนน validation ของโน้ตบุ๊ก
จึงมีข้อมูลรั่ว และผลบนคลิปนี้ก็ไม่ใช่การทดสอบกับคลิปที่โมเดลไม่เคยเห็น
งานนี้ใช้จุดอ้างอิงที่ทำไว้เพียง 72 จุดในสามเฟรม; ไม่มี annotation
ไข่ทุกฟอง, ตัวอย่างลบ หรือ ledger ของไข่แต่ละฟองที่ข้ามเส้นจริง
คะแนนแบบกรอบของ Kaggle กับแบบ mask ของ industoai ยังมีเกณฑ์จับจุดต่างกัน

**ยังไม่เปลี่ยนโมเดลใช้งานจริง** เพราะยังตัดสินจำนวนพลาด/นับซ้ำไม่ได้
ก่อนตัดสินควรทำ physical crossing ledger ของคลิปอื่นที่ไม่อยู่ในข้อมูลฝึก
และลองปรับ threshold ของ tracker แบบออฟไลน์ โดยดูทั้งไข่ที่กลับมาติดตามได้
และ false positive / ID switch ที่เพิ่มขึ้น

ไฟล์ Kaggle `best.pt` เก็บแยกใน `models/candidates/kaggle_egg_yolo11n.pt`
(SHA-256 ใน `models/candidates/kaggle_egg_yolo11n.json`), export OpenVINO
ใน `models/candidates/kaggle_egg_yolo11n_openvino_model/`.
ข้อกำหนดการใช้โมเดล/ไลบรารีควรตรวจอีกครั้งก่อนนำไปใช้งานเชิงพาณิชย์

## รันทดสอบซ้ำ

จากโฟลเดอร์โปรเจกต์ด้วย Python 3.12 ใน `.venv`:

```powershell
.\.venv\Scripts\python.exe -m backend.diagnose_pipeline --model current
.\.venv\Scripts\python.exe -m backend.diagnose_pipeline --model current640
.\.venv\Scripts\python.exe -m backend.diagnose_pipeline --model kaggle
```

แต่ละคำสั่งสร้างรายงาน JSON และภาพ audit แยกกัน ไม่เปิดระบบเว็บ
และไม่เขียน ledger ของการนับใน Dashboard
