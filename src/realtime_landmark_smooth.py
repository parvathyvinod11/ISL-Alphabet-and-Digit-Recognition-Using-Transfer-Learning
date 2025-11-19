# src/realtime_landmark_smooth_char_only.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

IMG_SIZE = 128

# ===============================
# Load trained model and classes
# ===============================
model = tf.keras.models.load_model("saved_models1/sign_model_unfreezed_full.h5")

with open("data/processed_with_v/classes.txt") as f:
    classes = [line.strip() for line in f]

# ===============================
# Mediapipe Hands setup
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Function to render landmarks
# ===============================
def render_landmark_image(landmarks, img_size=IMG_SIZE):
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255
    for i, lm in enumerate(landmarks):
        x = int(lm.x * (img_size - 1))
        y = int(lm.y * (img_size - 1))
        color = 0 if i < 21 else 100
        cv2.circle(canvas, (x, y), 2, color, -1)
    return canvas

# ===============================
# Webcam setup
# ===============================
cap = cv2.VideoCapture(0)

# Temporal smoothing buffer
frame_buffer = []
BUFFER_SIZE = 5
last_char = ""

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    char_pred = ""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Process hands
    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            all_landmarks.extend(hand_landmarks.landmark)

        # Render landmark image
        landmark_img = render_landmark_image(all_landmarks, IMG_SIZE)
        landmark_img = landmark_img.astype("float32") / 255.0
        landmark_img = np.expand_dims(landmark_img, axis=(0, -1))
        landmark_img = np.repeat(landmark_img, 3, axis=-1)

        # Predict sign
        pred = model.predict(landmark_img, verbose=0)
        class_id = np.argmax(pred)
        prob = np.max(pred)

        if prob > 0.5:
            char_pred = classes[class_id]

    # Temporal smoothing (optional, just for stability)
    if char_pred != "":
        frame_buffer.append(char_pred)
        if len(frame_buffer) > BUFFER_SIZE:
            frame_buffer.pop(0)

        if len(frame_buffer) == BUFFER_SIZE and all(c == char_pred for c in frame_buffer):
            last_char = char_pred
    else:
        frame_buffer = []
        last_char = ""

    # Combine webcam + skeleton canvas
    combined = np.hstack((frame, cv2.resize(canvas, (frame.shape[1], frame.shape[0]))))

    # Display only the character
    cv2.putText(combined, f"Character: {last_char}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("ISL Sign Language Detection", combined)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
