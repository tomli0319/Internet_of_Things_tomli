import time

import cv2
import numpy as np
from picamera2 import Picamera2

import config as cfg


def setup_camera():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    return picam2


def capture_bgr(picam2):
    frame_rgb = picam2.capture_array()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def create_detector():
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=cfg.BG_HISTORY,
        varThreshold=cfg.BG_VAR_THRESHOLD,
        detectShadows=cfg.BG_DETECT_SHADOWS,
    )
    kernel = np.ones((cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE), np.uint8)
    return subtractor, kernel


def apply_ignore_zone(mask):
    if cfg.USE_IGNORE_ZONE:
        x1, y1, x2, y2 = cfg.IGNORE_ZONE
        mask[y1:y2, x1:x2] = 0
    return mask


def detect(frame_bgr, subtractor, kernel):
    fg_mask = subtractor.apply(frame_bgr, learningRate=cfg.BG_LEARNING_RATE_IDLE)
    _, fg_mask = cv2.threshold(fg_mask, cfg.FG_THRESHOLD, 255, cv2.THRESH_BINARY)
    fg_mask = apply_ignore_zone(fg_mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.MIN_CONTOUR_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < cfg.MIN_CONTOUR_WIDTH_PX or h < cfg.MIN_CONTOUR_HEIGHT_PX:
            continue
        if area > best_area:
            best_area = area
            best_box = (x, y, w, h)

    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
    score = float(best_area) / frame_area if frame_area else 0.0
    detected = score >= cfg.MIN_OCCUPANCY_SCORE and best_box is not None
    if not detected:
        best_box = None
    return detected, score, best_box, fg_mask


def annotate(frame_bgr, box, score, detected):
    annotated = frame_bgr.copy()
    if cfg.DRAW_IGNORE_ZONE and cfg.USE_IGNORE_ZONE:
        x1, y1, x2, y2 = cfg.IGNORE_ZONE
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(annotated, "ignored zone", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"motion {score:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    status = "OBJECT IN FRAME" if detected else "CLEAR"
    cv2.putText(annotated, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, f"threshold={cfg.MIN_OCCUPANCY_SCORE:.3f}", (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def main():
    picam2 = setup_camera()
    subtractor, kernel = create_detector()

    print("Press q to quit. Press r to reset the background model.")
    print("This version defaults to full-frame detection with the ignore zone OFF.")

    for _ in range(cfg.BACKGROUND_WARMUP_FRAMES):
        frame_bgr = capture_bgr(picam2)
        subtractor.apply(frame_bgr, learningRate=cfg.BG_LEARNING_RATE_IDLE)
        time.sleep(0.03)

    try:
        while True:
            frame_bgr = capture_bgr(picam2)
            detected, score, box, fg_mask = detect(frame_bgr, subtractor, kernel)
            annotated = annotate(frame_bgr, box, score, detected)

            cv2.imshow("Full-Frame Motion Preview", annotated)
            cv2.imshow("Foreground Mask", fg_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                subtractor, kernel = create_detector()
                print("Background model reset.")

            time.sleep(cfg.IDLE_CHECK_INTERVAL_S)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
