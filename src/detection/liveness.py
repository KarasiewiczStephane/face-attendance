"""Liveness detection for anti-spoofing.

Implements blink detection using Eye Aspect Ratio (EAR) and texture
analysis using Local Binary Patterns (LBP) to distinguish live faces
from printed photos or screen displays.
"""

from collections import deque

import cv2
import numpy as np
from scipy.spatial import distance as dist

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BlinkDetector:
    """Detect eye blinks using Eye Aspect Ratio (EAR).

    Tracks EAR over consecutive frames to detect blink events,
    which indicate a live face rather than a static image.

    Args:
        ear_threshold: EAR value below which eyes are considered closed.
        consec_frames: Number of consecutive frames below threshold to count as blink.
        history_size: Size of the EAR history buffer.
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        consec_frames: int = 3,
        history_size: int = 30,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history: deque[float] = deque(maxlen=history_size)

    @staticmethod
    def eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """Compute Eye Aspect Ratio for a single eye.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            eye_points: Array of 6 landmark points for one eye.

        Returns:
            Eye aspect ratio value.
        """
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])
        h = dist.euclidean(eye_points[0], eye_points[3])

        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def process_frame(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
    ) -> tuple[float, bool]:
        """Process a frame and detect blink.

        Args:
            left_eye: 6 landmark points for left eye.
            right_eye: 6 landmark points for right eye.

        Returns:
            Tuple of (average_ear, blink_detected).
        """
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        self.ear_history.append(avg_ear)

        blink_detected = False
        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                blink_detected = True
            self.frame_counter = 0

        return avg_ear, blink_detected

    def get_blink_count(self) -> int:
        """Return total number of blinks detected.

        Returns:
            Blink count.
        """
        return self.blink_counter

    def reset(self) -> None:
        """Reset all blink detection state."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.ear_history.clear()


class TextureAnalyzer:
    """Analyze face texture using LBP to detect print attacks.

    Uses Local Binary Pattern histograms to distinguish real faces
    (high texture entropy) from printed photos (low texture entropy).

    Args:
        threshold: Minimum entropy score to consider face as live.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def compute_lbp(
        self,
        image: np.ndarray,
        radius: int = 1,
        neighbors: int = 8,
    ) -> np.ndarray:
        """Compute Local Binary Pattern for a grayscale image.

        Args:
            image: Input image (grayscale or BGR).
            radius: Radius of the circular LBP pattern.
            neighbors: Number of neighboring points.

        Returns:
            LBP encoded image.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                for n in range(neighbors):
                    angle = 2 * np.pi * n / neighbors
                    x = int(round(j + radius * np.cos(angle)))
                    y = int(round(i - radius * np.sin(angle)))
                    if image[y, x] >= center:
                        code |= 1 << n
                lbp[i - radius, j - radius] = code

        return lbp

    def compute_lbp_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized LBP histogram.

        Args:
            image: Input image.

        Returns:
            Normalized histogram of LBP values.
        """
        lbp = self.compute_lbp(image)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7
        return hist

    def analyze(self, face_image: np.ndarray) -> tuple[float, bool]:
        """Analyze if face is real or printed/displayed.

        Args:
            face_image: Cropped face image (BGR or grayscale).

        Returns:
            Tuple of (liveness_score, is_live).
        """
        hist = self.compute_lbp_histogram(face_image)

        # Live faces have more varied texture (higher entropy)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # Normalize to 0-1 range (max entropy for 256 bins is 8)
        liveness_score = float(entropy / 8.0)

        return liveness_score, liveness_score >= self.threshold


class LivenessDetector:
    """Combined liveness detection using blink and texture analysis.

    Args:
        blink_threshold: EAR threshold for blink detection.
        texture_threshold: Entropy threshold for texture analysis.
        required_blinks: Number of blinks required for blink-based liveness.
    """

    def __init__(
        self,
        blink_threshold: float = 0.25,
        texture_threshold: float = 0.5,
        required_blinks: int = 1,
    ) -> None:
        self.blink_detector = BlinkDetector(ear_threshold=blink_threshold)
        self.texture_analyzer = TextureAnalyzer(threshold=texture_threshold)
        self.required_blinks = required_blinks

    def check_liveness(
        self,
        face_image: np.ndarray,
        eye_landmarks: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict:
        """Check if a face is live.

        Args:
            face_image: Cropped face image.
            eye_landmarks: Optional tuple of (left_eye, right_eye) landmark arrays.

        Returns:
            Dict with texture_score, texture_live, blink_count,
            blink_live, overall_live, and confidence.
        """
        texture_score, texture_live = self.texture_analyzer.analyze(face_image)

        if eye_landmarks:
            left_eye, right_eye = eye_landmarks
            self.blink_detector.process_frame(left_eye, right_eye)

        blink_count = self.blink_detector.get_blink_count()
        blink_live = blink_count >= self.required_blinks

        # Texture alone can pass; blinks boost confidence
        overall_live = texture_live
        confidence = texture_score

        if blink_live:
            confidence = min(1.0, confidence + 0.2)

        return {
            "texture_score": texture_score,
            "texture_live": texture_live,
            "blink_count": blink_count,
            "blink_live": blink_live,
            "overall_live": overall_live,
            "confidence": confidence,
        }

    def reset(self) -> None:
        """Reset liveness detection state."""
        self.blink_detector.reset()
