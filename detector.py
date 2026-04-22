import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2

import config as cfg
from db import (
    get_live_feed_enabled,
    init_db,
    insert_event,
    update_live_frame,
    update_status,
)


class FullFrameMotionDetector:
    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.BG_HISTORY,
            varThreshold=cfg.BG_VAR_THRESHOLD,
            detectShadows=cfg.BG_DETECT_SHADOWS,
        )
        self.kernel = np.ones((cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE), np.uint8)

    def warmup(self, picam2):
        print(f"Warming background model with {cfg.BACKGROUND_WARMUP_FRAMES} frames...")
        for _ in range(cfg.BACKGROUND_WARMUP_FRAMES):
            frame_bgr = capture_bgr(picam2)
            self.subtractor.apply(frame_bgr, learningRate=cfg.BG_LEARNING_RATE_IDLE)
            time.sleep(0.03)
        print("Background model ready.")

    def _apply_ignore_zone(self, fg_mask):
        if cfg.USE_IGNORE_ZONE:
            x1, y1, x2, y2 = cfg.IGNORE_ZONE
            fg_mask[y1:y2, x1:x2] = 0
        return fg_mask

    def detect(self, frame_bgr, learning_rate):
        fg_mask = self.subtractor.apply(frame_bgr, learningRate=learning_rate)
        _, fg_mask = cv2.threshold(fg_mask, cfg.FG_THRESHOLD, 255, cv2.THRESH_BINARY)
        fg_mask = self._apply_ignore_zone(fg_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_area = 0.0
        detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.MIN_CONTOUR_AREA_PX:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < cfg.MIN_CONTOUR_WIDTH_PX or h < cfg.MIN_CONTOUR_HEIGHT_PX:
                continue
            if area > best_area:
                best_area = float(area)
                best_box = (x, y, w, h)

        frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
        occupancy_score = best_area / frame_area if frame_area else 0.0
        detected = occupancy_score >= cfg.MIN_OCCUPANCY_SCORE and best_box is not None
        if not detected:
            best_box = None
        return detected, occupancy_score, best_box, fg_mask


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


def annotate_frame(frame_bgr, best_box, occupancy_score, event_active):
    annotated = frame_bgr.copy()
    if cfg.DRAW_IGNORE_ZONE and cfg.USE_IGNORE_ZONE:
        x1, y1, x2, y2 = cfg.IGNORE_ZONE
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            "ignored zone",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )
    if best_box is not None:
        x, y, w, h = best_box
        color = (0, 255, 0) if event_active else (0, 191, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            f"motion {occupancy_score:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    cv2.putText(
        annotated,
        f"event_active={int(event_active)} threshold={cfg.MIN_OCCUPANCY_SCORE:.3f}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return annotated


def save_snapshot(frame_bgr, best_box, event_id, occupancy_score):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"event_{event_id}_{ts}.jpg"
    output_path = cfg.SNAPSHOT_DIR / filename
    annotated = annotate_frame(frame_bgr, best_box, occupancy_score, event_active=True)
    cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.JPEG_QUALITY])
    return f"snapshots/{filename}"


def save_live_frame(frame_bgr, best_box, occupancy_score, event_active):
    filename = "live.jpg"
    output_path = cfg.LIVE_DIR / filename
    annotated = annotate_frame(frame_bgr, best_box, occupancy_score, event_active)
    cv2.imwrite(str(output_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.JPEG_QUALITY])
    return f"live/{filename}"


def maybe_write_live_frame(frame_bgr, best_box, occupancy_score, event_active, last_write_ts):
    enabled = get_live_feed_enabled()
    now = time.time()
    if enabled and (now - last_write_ts >= cfg.LIVE_FEED_CAPTURE_INTERVAL_S):
        rel_path = save_live_frame(frame_bgr, best_box, occupancy_score, event_active)
        update_live_frame(rel_path)
        return now
    return last_write_ts


def confirm_event_start(picam2, motion_detector):
    positive_frames = 0
    best_score = 0.0
    best_frame = None
    best_box = None

    for _ in range(cfg.CONFIRM_FRAMES):
        frame_bgr = capture_bgr(picam2)
        detected, occupancy_score, local_box, _ = motion_detector.detect(
            frame_bgr, learning_rate=cfg.BG_LEARNING_RATE_EVENT
        )
        if detected:
            positive_frames += 1
            if occupancy_score >= best_score:
                best_score = occupancy_score
                best_frame = frame_bgr.copy()
                best_box = local_box
        time.sleep(cfg.ACTIVE_CHECK_INTERVAL_S)

    confirmed = positive_frames >= cfg.MIN_POSITIVE_FRAMES
    return confirmed, positive_frames, best_score, best_frame, best_box


def main():
    init_db()

    picam2 = setup_camera()
    motion_detector = FullFrameMotionDetector()
    motion_detector.warmup(picam2)

    event_active = False
    event_id = 0
    event_start_epoch = None
    event_start_iso = None
    snapshot_path = ""
    start_positive_frames = 0
    start_best_score = 0.0
    last_status_update = 0.0
    last_live_write = 0.0
    lost_windows = 0
    idle_trigger_windows = 0

    print("Detector started.")

    try:
        while True:
            frame_bgr = capture_bgr(picam2)

            if not event_active:
                occupied, occupancy_score, best_box, _ = motion_detector.detect(
                    frame_bgr, learning_rate=cfg.BG_LEARNING_RATE_IDLE
                )

                last_live_write = maybe_write_live_frame(
                    frame_bgr, best_box, occupancy_score, event_active, last_live_write
                )

                if time.time() - last_status_update >= cfg.STATUS_UPDATE_PERIOD_S:
                    update_status(
                        cat_detected=0,
                        cat_detected_drinking=0,
                        event_active=0,
                        current_event_id=event_id,
                        occupancy_score=0.0,
                        positive_frames=0,
                        event_start_time=None,
                        latest_snapshot=snapshot_path,
                    )
                    last_status_update = time.time()

                if occupied:
                    idle_trigger_windows += 1
                else:
                    idle_trigger_windows = 0

                if idle_trigger_windows >= cfg.IDLE_TRIGGER_WINDOWS:
                    print("Large moving object detected. Confirming event...")
                    confirmed, positive_frames, best_score, best_frame, best_box = confirm_event_start(
                        picam2, motion_detector
                    )
                    idle_trigger_windows = 0

                    if confirmed:
                        event_active = True
                        event_id += 1
                        event_start_epoch = time.time()
                        event_start_iso = datetime.now().isoformat(timespec="seconds")
                        start_positive_frames = positive_frames
                        start_best_score = best_score
                        lost_windows = 0
                        snapshot_path = (
                            save_snapshot(best_frame, best_box, event_id, best_score)
                            if best_frame is not None else ""
                        )

                        update_status(
                            cat_detected=1,
                            cat_detected_drinking=1,
                            event_active=1,
                            current_event_id=event_id,
                            occupancy_score=best_score,
                            positive_frames=positive_frames,
                            event_start_time=event_start_iso,
                            latest_snapshot=snapshot_path,
                        )
                        print(f"Event {event_id} started.")
                    else:
                        print("Motion rejected after confirmation window.")

                time.sleep(cfg.IDLE_CHECK_INTERVAL_S)

            else:
                occupied, occupancy_score, best_box, _ = motion_detector.detect(
                    frame_bgr, learning_rate=cfg.BG_LEARNING_RATE_EVENT
                )
                duration_s = int(time.time() - event_start_epoch)

                last_live_write = maybe_write_live_frame(
                    frame_bgr, best_box, occupancy_score, event_active, last_live_write
                )

                if time.time() - last_status_update >= cfg.STATUS_UPDATE_PERIOD_S:
                    update_status(
                        cat_detected=int(occupied),
                        cat_detected_drinking=1,
                        event_active=1,
                        current_event_id=event_id,
                        occupancy_score=occupancy_score,
                        positive_frames=start_positive_frames,
                        event_start_time=event_start_iso,
                        latest_snapshot=snapshot_path,
                    )
                    last_status_update = time.time()

                print(
                    f"Active | occupied={occupied} | duration={duration_s}s | occupancy={occupancy_score:.3f}"
                )

                if occupied:
                    lost_windows = 0
                else:
                    lost_windows += 1

                if lost_windows >= cfg.END_LOST_WINDOWS:
                    ended_at = datetime.now().isoformat(timespec="seconds")
                    insert_event(
                        event_id=event_id,
                        started_at=event_start_iso,
                        ended_at=ended_at,
                        duration_s=duration_s,
                        cat_detected=1,
                        cat_detected_drinking=1,
                        occupancy_score=max(start_best_score, occupancy_score),
                        positive_frames=start_positive_frames,
                        snapshot_path=snapshot_path,
                    )
                    update_status(
                        cat_detected=0,
                        cat_detected_drinking=0,
                        event_active=0,
                        current_event_id=event_id,
                        occupancy_score=0.0,
                        positive_frames=0,
                        event_start_time=None,
                        latest_snapshot=snapshot_path,
                    )
                    print(f"Event {event_id} ended. Duration = {duration_s}s")
                    event_active = False
                    event_start_epoch = None
                    event_start_iso = None
                    lost_windows = 0

                time.sleep(cfg.ACTIVE_CHECK_INTERVAL_S)

    except KeyboardInterrupt:
        print("Stopping detector...")
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()
