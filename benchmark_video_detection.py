import cv2
import torch
import os
from datetime import datetime
import numpy as np
import warnings

# Silenciar FutureWarnings de torch
warnings.filterwarnings("ignore", category=FutureWarning)

# Cargar modelo YOLOv5 preentrenado (coco)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cuda')
model.conf = 0.4  # confianza mínima

input_path = os.path.join('Videos', 'Video1.mp4')
output_path = os.path.join('Videos', 'Video1_annotated.mp4')

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"No se pudo abrir el video: {input_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Procesando video: {input_path}")
print(f"Guardando resultado en: {output_path}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------- Detección de objetos --------
    results = model(frame)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    object_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # máscara blanca

    for i in range(len(labels)):
        row = cords[i]
        if row[4] >= model.conf:
            x1, y1, x2, y2 = int(row[0]*width), int(row[1]*height), int(row[2]*width), int(row[3]*height)
            class_id = int(labels[i])
            name = model.names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{name}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Añadir zona negra en la máscara
            cv2.rectangle(object_mask, (x1, y1), (x2, y2), 0, -1)

    # -------- Detección de líneas de carretera --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Máscara: solo parte inferior
    roi_mask = np.zeros_like(edges)
    polygon = np.array([[  # Puedes ajustar el 0.45 para más o menos altura
        (0, height),
        (0, int(height * 0.45)),
        (width, int(height * 0.45)),
        (width, height)
    ]], np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)

    # Combinar la máscara del ROI con la de objetos
    combined_mask = cv2.bitwise_and(roi_mask, object_mask)
    edges = cv2.bitwise_and(edges, combined_mask)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=150)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("✅ Video procesado y guardado.")
