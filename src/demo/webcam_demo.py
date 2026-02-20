"""Real-time webcam face recognition demo.

Provides a live camera feed with face detection, recognition,
liveness checks, and automatic attendance logging overlaid on
the video stream.
"""

import argparse
from datetime import datetime

import cv2
import numpy as np

from ..database.attendance_db import AttendanceDatabase
from ..database.face_db import FaceDatabase
from ..detection import FaceProcessor
from ..detection.liveness import LivenessDetector
from ..matching import MatchingService
from ..utils.config import get_settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class WebcamDemo:
    """Real-time webcam face recognition demo.

    Captures video from a camera, detects and recognizes faces,
    performs liveness checks, and logs attendance in real time.

    Args:
        config_path: Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        settings = get_settings()
        self.config = settings.load_config()

        db_path = self.config["database"]["path"]

        self.face_processor = FaceProcessor(self.config)
        self.face_db = FaceDatabase(db_path)
        self.attendance_db = AttendanceDatabase(
            db_path,
            dedup_hours=self.config["attendance"]["dedup_window_hours"],
        )
        self.matching_service = MatchingService(
            self.face_db,
            threshold=self.config["matching"]["similarity_threshold"],
        )
        self.liveness_detector = LivenessDetector(
            blink_threshold=self.config["liveness"]["blink_threshold"],
            texture_threshold=self.config["liveness"]["texture_threshold"],
        )

    def draw_results(
        self,
        frame: np.ndarray,
        bbox: np.ndarray | None,
        name: str | None,
        confidence: float | None,
        liveness: dict | None,
        logged: bool,
    ) -> np.ndarray:
        """Draw bounding box, name, and status on the frame.

        Args:
            frame: Video frame (BGR).
            bbox: Face bounding box [x1, y1, x2, y2].
            name: Recognized person name.
            confidence: Match confidence score.
            liveness: Liveness detection result dict.
            logged: Whether attendance was logged this frame.

        Returns:
            Annotated frame.
        """
        if bbox is None:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

        x1, y1, x2, y2 = bbox.astype(int)

        if name:
            color = (0, 255, 0)
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if liveness:
            liveness_text = f"Liveness: {liveness['confidence']:.2f}"
            liveness_color = (0, 255, 0) if liveness["overall_live"] else (0, 0, 255)
            cv2.putText(
                frame,
                liveness_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                liveness_color,
                1,
            )

        if logged:
            cv2.putText(
                frame,
                "LOGGED",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame

    def run(self, camera_id: int = 0) -> None:
        """Run the webcam demo loop.

        Args:
            camera_id: Camera device ID (default: 0).
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error("Could not open camera %d", camera_id)
            return

        logger.info("Webcam demo started. Press 'q' to quit, 'r' to reset liveness.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            embedding, bbox, prob = self.face_processor.process_image(rgb_frame)

            name = None
            confidence = None
            liveness = None
            logged = False

            if embedding is not None:
                liveness = self.liveness_detector.check_liveness(frame)

                if liveness["overall_live"]:
                    match = self.matching_service.identify(embedding)

                    if match:
                        person, confidence = match
                        name = person["name"]

                        config = self.config
                        status = self.attendance_db.determine_status(
                            datetime.now(),
                            work_start=config["attendance"]["work_start"],
                            late_threshold_minutes=config["attendance"]["late_threshold_minutes"],
                        )
                        attendance_id = self.attendance_db.log_attendance(
                            person["id"],
                            confidence,
                            liveness["confidence"],
                            status,
                        )
                        logged = attendance_id is not None

            frame = self.draw_results(frame, bbox, name, confidence, liveness, logged)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                timestamp,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Face Attendance Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.liveness_detector.reset()
                logger.info("Liveness detector reset")

        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Entry point for the webcam demo CLI."""
    parser = argparse.ArgumentParser(description="Face Attendance Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    demo = WebcamDemo(args.config)
    demo.run(args.camera)


if __name__ == "__main__":
    main()  # pragma: no cover
