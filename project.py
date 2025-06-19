import cv2
import time
import threading
import queue
import numpy as np
from modules.state import initialize_states
from modules.detection import yolo_to_deepsort
from modules.fsm import update_states
from modules.visualize import drawing
from modules.sort_tracker import track_with_sort
from modules.clean_bbox import rm_duplicate

MODEL_PATH = "picnic_5n_fp16.engine"
VIDEO_PATH = "input.mp4"
BUFFER_SIZE = 1

def yolo_worker(frame_q: queue.Queue, result_dict: dict):
    from modules.detect import TRTInfer 
    model = TRTInfer(MODEL_PATH)        

    states = initialize_states()

    while True:
        item = frame_q.get()
        if item is None:
            break

        frame, orig_fps = item

        raw_detections = model.infer(frame)
        detections = yolo_to_deepsort(raw_detections)

        det_person = rm_duplicate([d[:5] for d in detections if d[5] == "person"], 20, "max_conf")
        det_mat    = rm_duplicate([d[:5] for d in detections if d[5] == "mat"],    20, "max_conf")
        det_bottle = rm_duplicate([d[:5] for d in detections if d[5] == "bottle"], 20, "max_conf")

        trk_person = track_with_sort(det_person, "person")
        trk_mat    = track_with_sort(det_mat,    "mat")
        trk_bottle = track_with_sort(det_bottle, "bottle")

        person_bb = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_person]
        mat_bb    = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_mat]
        bottle_bb = [(tid, (x1, y1, x2, y2)) for x1, y1, x2, y2, tid in trk_bottle]

        update_states(states, person_bb, mat_bb, bottle_bb, orig_fps)

        result_dict["bboxes"] = (person_bb, mat_bb, bottle_bb)
        result_dict["states"] = states.copy()

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f" Cannot open video source: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps == 0 or np.isnan(orig_fps):
        orig_fps = 30
    frame_interval = 1.0 / orig_fps

    # 윈도우를 한 번만 생성하고 크기 조정
    window_name = "🧺 Picnic Trash Detection – Realtime"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    # 전체화면으로 표시하려면 아래를 대신 사용하세요:
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_q = queue.Queue(maxsize=BUFFER_SIZE)
    result_dict = {"bboxes": ([], [], []), "states": {}}

    worker = threading.Thread(target=yolo_worker, args=(frame_q, result_dict), daemon=True)
    worker.start()

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Video read failed.")
            break

        # 최신 프레임만 큐에 유지
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put((frame.copy(), orig_fps))

        person_bb, mat_bb, bottle_bb = result_dict.get("bboxes", ([], [], []))
        states = result_dict.get("states", {})
        drawing(frame, person_bb, mat_bb, bottle_bb, states)

        # 항상 같은 윈도우 이름 사용
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # 재생 속도 조절
        elapsed = time.time() - prev_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        prev_time = time.time()

    # 워커 스레드 종료
    frame_q.put(None)
    worker.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



