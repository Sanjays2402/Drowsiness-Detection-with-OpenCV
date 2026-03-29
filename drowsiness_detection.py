"""
Drowsiness Detection System using OpenCV and face_recognition.

Monitors a driver's eyes in real-time via webcam and triggers an audio alarm
when the Eye Aspect Ratio (EAR) drops below a threshold for a sustained
number of consecutive frames — indicating drowsiness.

Published in IEEE: https://ieeexplore.ieee.org/document/9532758
Author: Sanjay Santhanam
"""

import argparse
import sys
import time
from typing import List, Tuple

import cv2
import face_recognition
import imutils
import numpy as np
from imutils.video import VideoStream
from playsound import playsound
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye: List[Tuple[int, int]]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    A high EAR means the eye is open; a low EAR means it is closed.

    Args:
        eye: List of 6 (x, y) landmark coordinates for one eye.

    Returns:
        The eye aspect ratio as a float.
    """
    vertical_a = dist.euclidean(eye[1], eye[5])
    vertical_b = dist.euclidean(eye[2], eye[4])
    horizontal = dist.euclidean(eye[0], eye[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def play_alarm(sound_path: str = "assets/alert1.mp3") -> None:
    """Play the drowsiness alert sound."""
    playsound(sound_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time drowsiness detection using Eye Aspect Ratio (EAR)."
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.25,
        help="EAR below this value is considered a closed eye (default: 0.25)",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=60,
        help="Consecutive frames below threshold before alarm triggers (default: 60)",
    )
    parser.add_argument(
        "--video-source",
        type=int,
        default=0,
        help="Video source index, e.g. 0 for default webcam (default: 0)",
    )
    parser.add_argument(
        "--alarm-sound",
        type=str,
        default="assets/alert1.mp3",
        help="Path to the alarm sound file (default: assets/alert1.mp3)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the drowsiness detection loop."""
    args = parse_args()

    ear_threshold: float = args.ear_threshold
    frame_count_limit: int = args.frame_count
    alarm_sound: str = args.alarm_sound

    print("[INFO] Starting video stream...")
    vs = VideoStream(src=args.video_source).start()
    time.sleep(1.0)  # Allow camera sensor to warm up

    consecutive_frames: int = 0
    ear: float = 0.0

    try:
        while True:
            frame = vs.read()
            if frame is None:
                break

            frame = imutils.resize(frame, width=450)

            # Detect facial landmarks
            face_landmarks_list = face_recognition.face_landmarks(frame)

            for landmarks in face_landmarks_list:
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]

                # Draw eye contours
                cv2.polylines(frame, [np.array(left_eye)], True, (100, 250, 0), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (100, 250, 0), 1)

                # Compute average EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < ear_threshold:
                    consecutive_frames += 1
                    if consecutive_frames >= frame_count_limit:
                        play_alarm(alarm_sound)
                        cv2.putText(
                            frame, "DROWSINESS ALERT!",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2,
                        )
                else:
                    consecutive_frames = 0

            # Display EAR on frame
            cv2.putText(
                frame, f"EAR: {ear:.2f}",
                (310, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2,
            )

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    main()
