"""
Human Tracking & Face Recognition - Demo
Simplified version for educational purposes.
"""

import cv2
import numpy as np
import argparse
import json
import os
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# ---- Configuration ----
CONFIG = {
    "yolo_model": "yolov8n.pt",
    "face_model": "buffalo_sc",
    "t_high": 0.65,
    "t_low": 0.40,
    "face_min_size": 30,
    "face_min_sharpness": 25.0,
    "save_dir": "output",
}


def compute_sharpness(crop):
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main(video_path, output_path):
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # Load models
    print("[INFO] Loading models...")
    yolo = YOLO(CONFIG["yolo_model"])
    face_app = FaceAnalysis(name=CONFIG["face_model"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )
    
    # State
    identities = {}       # {iid: {"embeddings": [], "tracks": []}}
    next_id = 1
    track_to_id = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Track persons
        results = yolo.track(
            frame, persist=True, classes=[0],
            tracker="botsort.yaml", verbose=False
        )[0]
        
        if results.boxes is None:
            out.write(frame)
            continue
        
        for box in results.boxes:
            if box.id is None:
                continue
            tid = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Face detection
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            faces = face_app.get(person_crop)
            if len(faces) == 0:
                continue
            
            # Best face by size
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            
            # Quality check
            if (fx2 - fx1) < CONFIG["face_min_size"]:
                continue
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            if compute_sharpness(face_crop) < CONFIG["face_min_sharpness"]:
                continue
            
            # Match identity
            emb = face.embedding / np.linalg.norm(face.embedding)
            assigned_id = None
            
            if identities:
                all_embs, all_ids = [], []
                for iid, data in identities.items():
                    for e in data["embeddings"]:
                        all_embs.append(e)
                        all_ids.append(iid)
                
                sims = cosine_similarity([emb], np.array(all_embs))[0]
                best = int(np.argmax(sims))
                if sims[best] >= CONFIG["t_high"]:
                    assigned_id = all_ids[best]
                    identities[assigned_id]["embeddings"].append(emb)
            
            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
                identities[assigned_id] = {"embeddings": [emb], "tracks": [tid]}
            
            track_to_id[tid] = assigned_id
            
            # Draw
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{assigned_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        out.write(frame)
        cv2.imshow("Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="output.mp4")
    args = parser.parse_args()
    main(args.video, args.output)