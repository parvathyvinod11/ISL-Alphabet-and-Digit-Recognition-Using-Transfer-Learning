# src/collect.py
import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

SAVE_DIR = "data/raw"  # inside subfolders for labels
IMG_SIZE = 128

def normalize_landmarks(landmarks):
    # landmarks: list of (x,y,z) normalized by image width/height
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    # translate so x,y center is zero, scale by max range
    coords = arr.reshape(-1,3)
    xy = coords[:, :2]
    mean = xy.mean(axis=0)
    xy_centered = xy - mean
    max_val = np.max(np.abs(xy_centered)) + 1e-6
    xy_norm = xy_centered / max_val
    coords[:, :2] = xy_norm
    return coords.flatten()

def render_landmark_image(landmarks, img_size=IMG_SIZE):
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255
    # draw landmark connections in black
    for lm in landmarks:
        x = int(lm.x * (img_size-1))
        y = int(lm.y * (img_size-1))
        cv2.circle(canvas, (x,y), radius=2, color=0, thickness=-1)
    return canvas

def collect(label):
    out_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    print("Press 's' to save sample, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Label: {label}  Saved: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("collect", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            if res.multi_hand_landmarks:
                # for now, take first hand or combine both
                all_lms = []
                for hand_landmarks in res.multi_hand_landmarks:
                    all_lms.extend(hand_landmarks.landmark)
                # If only one hand but target is two-handed sign, you may want to prompt
                vec = normalize_landmarks(all_lms)
                img = render_landmark_image(all_lms, IMG_SIZE)
                ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
                np.save(os.path.join(out_dir, f"{label}_{ts}_landmark.npy"), vec)
                cv2.imwrite(os.path.join(out_dir, f"{label}_{ts}.png"), img)
                count += 1
                print("saved", count)
            else:
                print("No hand detected. Try again.")
        elif k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "Q"
    collect(label)
