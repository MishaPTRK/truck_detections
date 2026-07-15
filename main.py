import csv
import os
from datetime import datetime

import cv2
import pytesseract
from collections import Counter, defaultdict, deque
from ultralytics import YOLO

# Якщо на Windows tesseract не в PATH, розкоментуй і вкажи свій шлях:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

truck_model = YOLO("yolo11n.pt")

# Окрема YOLO-модель, натренована для детекції номерних знаків
# Завантажити: https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt
plate_model = YOLO("license_plate_detector.pt")

truck_id_class = [k for k, v in truck_model.names.items() if v == "truck"][0]

CSV_PATH = "trucks_log.csv"


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


# Буфер голосування ОКРЕМО НА КОЖЕН track_id вантажівки
MIN_PLATE_LEN = 5      # коротші за це - явний брак (відблиск/погана детекція), відкидаємо одразу
MIN_SUBSTRING_LEN = 7  # мінімальна довжина кандидата на "справжній" номер (підбери під формат своєї країни)
MIN_READINGS_TO_TRUST = 8  # скільки читань зібрати перед тим, як довіряти результату

plate_history_by_id = defaultdict(lambda: deque(maxlen=40))
logged_ids = set()  # track_id, для яких вже записали рядок у CSV (щоб не дублювати)


def get_stable_plate(track_id, new_text):
    """
    Додає новий OCR-результат у буфер САМЕ ЦЬОГО track_id і повертає
    найімовірніший номер (або None, якщо ще недостатньо даних).

    Метод: серед усіх підпослідовностей довжини >= MIN_SUBSTRING_LEN з усіх
    зібраних кадрів ЦІЄЇ вантажівки шукаємо найчастішу.
    """
    history = plate_history_by_id[track_id]

    if new_text and len(new_text) >= MIN_PLATE_LEN:
        history.append(new_text)

    if len(history) < MIN_READINGS_TO_TRUST:
        return None

    substring_counts = Counter()
    for text in history:
        n = len(text)
        for length in range(MIN_SUBSTRING_LEN, n + 1):
            for start in range(0, n - length + 1):
                substring_counts[text[start:start + length]] += 1

    if not substring_counts:
        return None

    best_candidate = max(substring_counts.items(), key=lambda kv: (kv[1], len(kv[0])))
    return best_candidate[0]


def log_to_csv(track_id, plate_text):
    """Дописує рядок ID - номер - час у CSV (створює файл із заголовком, якщо його ще нема)."""
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "plate", "timestamp"])
        writer.writerow([track_id, plate_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    print(f"[ЗАПИСАНО В CSV] ID={track_id}  номер={plate_text}")


def process_frame(frame):
    """Обробляє один кадр: трекінг truck -> номер -> OCR -> голосування по ID -> CSV."""
    # persist=True - модель пам'ятає треки між кадрами і видає стабільні id
    results = truck_model.track(
        frame, classes=[truck_id_class], conf=0.65, persist=True, verbose=False
    )
    annotated = results[0].plot()

    boxes = results[0].boxes
    if boxes.id is None:
        # трекер ще не встиг призначити id (перші кадри) - пропускаємо
        return annotated

    for box, track_id_tensor in zip(boxes, boxes.id):
        track_id = int(track_id_tensor)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = frame[y1:y2, x1:x2]

        plate_box = find_plate_box(cropped_img)
        if plate_box is None:
            continue

        px1, py1, px2, py2, plate_conf = plate_box
        abs_x1, abs_y1 = x1 + px1, y1 + py1
        abs_x2, abs_y2 = x1 + px2, y1 + py2

        plate_crop = frame[abs_y1:abs_y2, abs_x1:abs_x2]
        raw_text = recognize_plate_text(plate_crop)
        stable_text = get_stable_plate(track_id, raw_text)

        print(f"ID={track_id}  Кадр: {raw_text}  |  Стабільний номер: {stable_text}  (conf: {plate_conf:.2f})")

        # Як тільки для цього ID вперше визначився стабільний номер - пишемо в CSV один раз
        if stable_text is not None and track_id not in logged_ids:
            log_to_csv(track_id, stable_text)
            logged_ids.add(track_id)

        cv2.rectangle(annotated, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
        label = f"ID{track_id}: {stable_text if stable_text else raw_text or '...'}"
        cv2.putText(
            annotated,
            label,
            (abs_x1, max(abs_y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return annotated


# Джерело відео:
# 0                                    - веб-камера
# "video.mp4"                         - відеофайл
# "rtsp://user:pass@192.168.1.90/..." - RTSP-потік з IP-камери
video_source = "m.mp4"

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    raise RuntimeError(f"Не вдалося відкрити джерело відео: {video_source}")

print("Старт обробки. Натисни 'q' для виходу.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Кадр не отримано (кінець відео або обрив зв'язку).")
        break

    annotated = process_frame(frame)

    cv2.imshow("Trucks", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()