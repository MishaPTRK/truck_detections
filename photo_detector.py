import cv2
import pytesseract
from ultralytics import YOLO

# Якщо на Windows tesseract не в PATH, розкоментуй і вкажи свій шлях:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

truck_model = YOLO("yolo11n.pt")

# Окрема YOLO-модель, натренована для детекції номерних знаків
# Завантажити: https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt
plate_model = YOLO("license_plate_detector.pt")

truck_id = [k for k, v in truck_model.names.items() if v == "truck"][0]


def find_plate_box(cropped_img, conf=0.45):
    """Повертає (x1, y1, x2, y2, conf) номера відносно cropped_img, або None."""
    if cropped_img is None or cropped_img.size == 0:
        return None

    results = plate_model.predict(cropped_img, conf=conf, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    return x1, y1, x2, y2, float(best_box.conf[0])


def recognize_plate_text(plate_img):
    """Приймає кроп номерної пластини, повертає розпізнаний текст (str) або None."""
    if plate_img is None or plate_img.size == 0:
        return None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    text = "".join(ch for ch in text if ch.isalnum()).upper()
    return text if text else None


image_path = r"C:\Users\User\Downloads\Знімок екрана 2026-07-10 155459.png"
frame = cv2.imread(image_path)

results = truck_model.predict(frame, classes=[truck_id], conf=0.65, verbose=False)
annotated = results[0].plot()

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_img = frame[y1:y2, x1:x2]

    plate_box = find_plate_box(cropped_img)
    if plate_box is None:
        continue

    px1, py1, px2, py2, plate_conf = plate_box
    abs_x1, abs_y1 = x1 + px1, y1 + py1
    abs_x2, abs_y2 = x1 + px2, y1 + py2

    plate_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
    plate_text = recognize_plate_text(plate_crop)

    # Вивід номера в чат/консоль
    print(f"Знайдено номер: {plate_text} (впевненість детекції: {plate_conf:.2f})")

    cv2.rectangle(annotated, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
    label = plate_text if plate_text else "?"
    cv2.putText(
        annotated,
        label,
        (abs_x1, max(abs_y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

cv2.imshow("Trucks", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()