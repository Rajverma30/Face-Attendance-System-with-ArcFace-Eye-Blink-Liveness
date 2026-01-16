import os
import cv2
import time
import csv
import pickle
import numpy as np
from datetime import datetime, date
import onnxruntime as ort
from insightface.app import FaceAnalysis

try:
    from insightface.utils import face_align
except Exception:
    face_align = None

# ---------- CONFIG ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARC_FACE_ONNX_PATH = os.path.join(PROJECT_ROOT, r"models\antelopev2\arcface.onnx")

PHOTOS_ROOT = os.path.join(PROJECT_ROOT, "photos")
FACE_DB_PATH = os.path.join(PROJECT_ROOT, "face_db.pkl")
ATTENDANCE_CSV = os.path.join(PROJECT_ROOT, "attendance.csv")

COSINE_THRESHOLD = 0.65
ATTENDANCE_THRESHOLD = 0.70
MIN_FACE_SIZE = 60
EAR_THRESH = 0.35
EAR_CONSEC_FRAMES = 1
EAR_DELTA = 0.03  # mark if EAR changes by this much between frames


# ---------- UTILS ----------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


def ensure_files():
    if not os.path.exists(ARC_FACE_ONNX_PATH):
        raise FileNotFoundError(f"ArcFace ONNX not found at {ARC_FACE_ONNX_PATH}")
    os.makedirs(PHOTOS_ROOT, exist_ok=True)
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["NAME", "DATE", "TIME"])  # header


def load_face_db():
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_face_db(db):
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(db, f)


def append_attendance(name: str):
    today = date.today().isoformat()
    nowt = datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, nowt])


def today_present_names() -> set:
    today = date.today().isoformat()
    if not os.path.exists(ATTENDANCE_CSV):
        return set()
    pres = set()
    with open(ATTENDANCE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("DATE") == today:
                pres.add(r.get("NAME", ""))
    return pres


# ---------- MODELS ----------
det_app = FaceAnalysis(providers=["CPUExecutionProvider"])
det_app.prepare(ctx_id=-1, det_size=(640, 640))

sess = ort.InferenceSession(ARC_FACE_ONNX_PATH, providers=["CPUExecutionProvider"])
arc_in = sess.get_inputs()[0].name
arc_out = sess.get_outputs()[0].name


def preprocess_arcface_crop(img_bgr, bbox, landmark=None):
    aligned_rgb = None
    if landmark is not None and face_align is not None:
        try:
            aligned_rgb = face_align.norm_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), landmark, image_size=112)
        except Exception:
            aligned_rgb = None
    if aligned_rgb is None:
        x1, y1, x2, y2 = map(int, bbox)
        face = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if face.size == 0:
            return None
        aligned_rgb = cv2.cvtColor(cv2.resize(face, (112, 112)), cv2.COLOR_BGR2RGB)
    face = aligned_rgb.astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, 0).astype(np.float32)
    return face


def aligned_face_bgr(img_bgr, bbox, landmark=None):
    if landmark is not None and face_align is not None:
        try:
            aligned_rgb = face_align.norm_crop(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), landmark, image_size=112)
            return cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    x1, y1, x2, y2 = map(int, bbox)
    face = img_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if face.size == 0:
        return None
    return cv2.resize(face, (112, 112))


def embed_arcface(img_bgr, bbox, landmark=None):
    inp = preprocess_arcface_crop(img_bgr, bbox, landmark)
    if inp is None:
        return None
    out = sess.run([arc_out], {arc_in: inp})
    emb = np.array(out[0]).reshape(-1)
    return l2_normalize(emb)


def embed_from_aligned_112(img_bgr_112):
    # Directly embed an already aligned 112x112 BGR image
    if img_bgr_112 is None or img_bgr_112.size == 0:
        return None
    if img_bgr_112.shape[0] != 112 or img_bgr_112.shape[1] != 112:
        try:
            img_bgr_112 = cv2.resize(img_bgr_112, (112, 112))
        except Exception:
            return None
    face = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, 0).astype(np.float32)
    out = sess.run([arc_out], {arc_in: face})
    emb = np.array(out[0]).reshape(-1)
    return l2_normalize(emb)


def largest_face(faces):
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0]


def enroll_from_folder(root):
    db = {}
    if not os.path.isdir(root):
        return db
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for fn in os.listdir(pdir):
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                continue
            faces = det_app.get(img)
            f = largest_face(faces)
            if f is None:
                # If this is an already aligned crop (112x112), try direct embedding
                if img.shape[0] == 112 and img.shape[1] == 112:
                    emb = embed_from_aligned_112(img)
                else:
                    emb = None
            else:
                emb = embed_arcface(img, f.bbox, getattr(f, "kps", None))
            if emb is not None:
                embs.append(emb)
        if embs:
            db[person] = l2_normalize(np.mean(np.stack(embs), axis=0))
            print(f"[INFO] Enrolled {person} with {len(embs)} images")
    return db


def build_or_update_face_db():
    db = load_face_db()
    folder_db = enroll_from_folder(PHOTOS_ROOT)
    db.update(folder_db)
    if db:
        save_face_db(db)
        print(f"[OK] Saved {len(db)} identities to {FACE_DB_PATH}")
    else:
        print("[WARN] No identities enrolled yet. Add images in photos/<Name>.")
    return db


def best_match(emb, db: dict):
    emb = emb.reshape(-1)
    best_name, best_sim = None, -1
    for name, ref in db.items():
        ref = ref.reshape(-1)
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < COSINE_THRESHOLD:
        return "Unknown", best_sim, False
    return best_name, best_sim, True


def eye_points_from_bbox(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None, None
    left_cx = x1 + int(0.35 * w)
    right_cx = x1 + int(0.65 * w)
    cy = y1 + int(0.40 * h)
    ew = int(0.12 * w)
    eh = int(0.06 * h)
    def six(cx, cy, ew, eh):
        return [
            (cx - ew, cy),
            (cx - ew//2, cy - eh),
            (cx + ew//2, cy - eh),
            (cx + ew, cy),
            (cx + ew//2, cy + eh),
            (cx - ew//2, cy + eh)
        ]
    return six(left_cx, cy, ew, eh), six(right_cx, cy, ew, eh)


def ear(p):
    p = np.array(p)
    if p.shape[0] < 6:
        return 0.0
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C)


def enroll_via_webcam(num_samples=20, min_interval_sec=0.25):
    name = input("Enter new student name: ").strip()
    if not name:
        print("[WARN] Empty name. Aborting.")
        return
    save_dir = os.path.join(PHOTOS_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Capturing {num_samples} samples for {name}. Press 'q' to cancel.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    captured = 0
    last_ts = 0.0
    try:
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame")
                break
            faces = det_app.get(frame)
            f = largest_face(faces)
            if f is not None:
                now_ts = time.time()
                if now_ts - last_ts >= min_interval_sec:
                    face_img = aligned_face_bgr(frame, f.bbox, getattr(f, "kps", None))
                    if face_img is not None and face_img.size != 0:
                        out_path = os.path.join(save_dir, f"{name}_{captured:03d}.jpg")
                        cv2.imwrite(out_path, face_img)
                        captured += 1
                        last_ts = now_ts
                        cv2.putText(frame, f"Saved {captured}/{num_samples}", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
            cv2.imshow("Enroll - press q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Enrollment cancelled by user.")
                break
        print(f"[INFO] Captured {captured} images for {name}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured > 0:
        db = build_or_update_face_db()
        if name in db:
            print(f"[OK] {name} enrolled successfully.")
        else:
            print(f"[WARN] Enrollment may have failed. Check images in {save_dir}.")


def run_webcam():
    db = load_face_db()
    if not db:
        print("[WARN] face_db.pkl empty. Building from photos/ ...")
        db = build_or_update_face_db()
        if not db:
            print("[ERROR] No enrolled faces. Exiting.")
            return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    print("Webcam open. Press 'q' to quit.")

    blink_state = {}
    marked_today = set()
    mark_message_until = {}  # name -> timestamp until when to show message

    stop_after_mark = False
    try:
        while True:
            if stop_after_mark:
                break
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Could not read frame")
                break

            faces = det_app.get(frame)
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                    continue
                emb = embed_arcface(frame, f.bbox, getattr(f, "kps", None))
                if emb is None:
                    continue
                name, sim, recognized = best_match(emb, db)

                le, re = eye_points_from_bbox((x1, y1, x2, y2))
                if le is None or re is None:
                    ear_val = 1.0
                else:
                    ear_val = (ear(le) + ear(re)) / 2.0

                st = blink_state.get(name, {"counter": 0, "last_ear": ear_val})
                if ear_val < EAR_THRESH:
                    st["counter"] += 1
                    # Trigger on reaching threshold frames while eyes closed
                    if (
                        st["counter"] == EAR_CONSEC_FRAMES and recognized and name != "Unknown"
                    ):
                        key = (name, date.today().isoformat())
                        if key not in marked_today:
                            append_attendance(name)
                            marked_today.add(key)
                            mark_message_until[name] = time.time() + 2.0
                            print(f"Attendance marked for {name}")
                            cv2.putText(frame, f"Attendance marked for {name}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
                            # Show confirmation briefly then stop webcam
                            cv2.imshow("ArcFace (ONNX) + Blink Attendance", frame)
                            cv2.waitKey(300)
                            stop_after_mark = True
                else:
                    st["counter"] = 0

                # Also trigger if EAR changes sharply between frames (small eye movement)
                if not stop_after_mark:
                    last_ear = st.get("last_ear", ear_val)
                    if abs(ear_val - last_ear) >= EAR_DELTA and recognized and name != "Unknown":
                        key = (name, date.today().isoformat())
                        if key not in marked_today:
                            append_attendance(name)
                            marked_today.add(key)
                            mark_message_until[name] = time.time() + 2.0
                            print(f"Attendance marked for {name}")
                            cv2.putText(frame, f"Attendance marked for {name}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)
                            cv2.imshow("ArcFace (ONNX) + Blink Attendance", frame)
                            cv2.waitKey(300)
                            stop_after_mark = True

                st["last_ear"] = ear_val
                blink_state[name] = st

                color = (0, 255, 0) if recognized else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} sim={sim:.2f} EAR={ear_val:.2f}",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # Show confirmation message for 2 seconds after marking
                if name in mark_message_until and time.time() < mark_message_until[name]:
                    cv2.putText(frame, f"Attendance marked for {name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0), 2)

            cv2.imshow("ArcFace (ONNX) + Blink Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


def show_attendance_today():
    db = load_face_db()
    enrolled = set(db.keys())
    present = today_present_names()
    absent = sorted(enrolled - present)
    present_sorted = sorted(present)

    print(f"\n--- Attendance for {date.today().isoformat()} ---")
    print(f"Present ({len(present_sorted)}):")
    if present_sorted:
        for n in present_sorted:
            print(f"- {n}")
    else:
        print("- None")
    print(f"Absent ({len(absent)}):")
    if absent:
        for n in absent:
            print(f"- {n}")
    else:
        print("- None")


def enroll_student():
    print("1) Enroll via webcam  2) Enroll from photos folder")
    mode = input("Choose (1/2): ").strip()
    if mode == "1":
        enroll_via_webcam()
    else:
        name = input("Enter student name: ").strip()
        folder = os.path.join(PHOTOS_ROOT, name)
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Place images of {name} in {folder}.")
        input("Press Enter after adding images...")
        db = build_or_update_face_db()
        if name in db:
            print(f"[INFO] {name} enrolled successfully.")
        else:
            print(f"[WARN] Enrollment failed. Check images.")


def purge_student():
    name = input("Enter student name to purge: ").strip()
    if not name:
        print("[WARN] Empty name. Aborting.")
        return
    folder = os.path.join(PHOTOS_ROOT, name)
    if os.path.isdir(folder):
        try:
            for fn in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, fn))
                except Exception:
                    pass
            os.rmdir(folder)
            print(f"[OK] Deleted photos folder: {folder}")
        except Exception as e:
            print(f"[WARN] Could not delete folder {folder}: {e}")
    else:
        print(f"[INFO] Photos folder not found: {folder}")

    db = load_face_db()
    if name in db:
        db.pop(name, None)
        save_face_db(db)
        print(f"[OK] Removed {name} from {FACE_DB_PATH}")
    else:
        print(f"[INFO] {name} not found in {FACE_DB_PATH}")

    # Remove entries from CSV
    if os.path.exists(ATTENDANCE_CSV):
        rows = []
        with open(ATTENDANCE_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("NAME") != name:
                    rows.append(r)
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["NAME", "DATE", "TIME"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Removed attendance rows for {name}")


def show_menu():
    print("\n===== Face Attendance (File-based) =====")
    print("1. Enroll new student")
    print("2. Mark attendance (webcam)")
    print("3. Show today's attendance list")
    print("4. Purge a student's data")
    print("5. Exit")


def main():
    ensure_files()
    while True:
        show_menu()
        choice = input("Enter choice (1/2/3/4/5): ").strip()
        if choice == "1":
            enroll_student()
        elif choice == "2":
            run_webcam()
        elif choice == "3":
            show_attendance_today()
        elif choice == "4":
            purge_student()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()


