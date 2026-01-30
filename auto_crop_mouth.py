
import cv2
import mediapipe as mp # type: ignore
import os

IMAGE_DIR = "images"
OUTPUT_DIR = "cropped_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# Mouth landmark indices
MOUTH_LANDMARKS = [61, 291, 78, 308, 13, 14, 17, 0]

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    if not result.multi_face_landmarks:
        continue

    landmarks = result.multi_face_landmarks[0].landmark

    xs, ys = [], []
    for idx in MOUTH_LANDMARKS:
        xs.append(int(landmarks[idx].x * w))
        ys.append(int(landmarks[idx].y * h))

    x1, x2 = max(0, min(xs)-30), min(w, max(xs)+30)
    y1, y2 = max(0, min(ys)-30), min(h, max(ys)+30)

    mouth_crop = image[y1:y2, x1:x2]

    if mouth_crop.size == 0:
        continue

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), mouth_crop)

print("✅ Mouth cropping completed")
