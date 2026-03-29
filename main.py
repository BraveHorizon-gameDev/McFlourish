import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("fire_detector_model.h5")

classes = ["fire", "no_fire", "smoke"]

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    class_id = np.argmax(pred)
    confidence = np.max(pred)

    label = classes[class_id]

    # DEBUG (IMPORTANT)
    print(pred)

    if label == "fire":
        text = f"FIRE {confidence:.2f}"
        color = (0, 0, 255)
    elif label == "smoke":
        text = f"SMOKE {confidence:.2f}"
        color = (0, 255, 255)
    else:
        text = f"NO FIRE {confidence:.2f}"
        color = (0, 255, 0)

    cv.putText(frame, text, (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv.imshow("Fire Detection", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()