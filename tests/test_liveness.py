"""Tests for liveness detection (anti-spoofing)."""

import numpy as np
import pytest

from src.detection.liveness import (
    BlinkDetector,
    LivenessDetector,
    TextureAnalyzer,
)


def _make_eye_landmarks(open_ratio: float = 0.3) -> np.ndarray:
    """Generate synthetic eye landmarks with controlled openness.

    Args:
        open_ratio: Vertical to horizontal ratio (0 = closed, 0.5 = wide open).

    Returns:
        Array of 6 (x, y) landmark points.
    """
    # Horizontal span
    h_span = 20.0
    v_span = h_span * open_ratio

    return np.array(
        [
            [0, 0],  # p1: left corner
            [h_span * 0.33, -v_span],  # p2: upper-left
            [h_span * 0.66, -v_span],  # p3: upper-right
            [h_span, 0],  # p4: right corner
            [h_span * 0.66, v_span],  # p5: lower-right
            [h_span * 0.33, v_span],  # p6: lower-left
        ],
        dtype=np.float64,
    )


class TestBlinkDetector:
    """Tests for BlinkDetector class."""

    def test_eye_aspect_ratio_open(self) -> None:
        """Test EAR for open eye is above threshold."""
        open_eye = _make_eye_landmarks(open_ratio=0.3)
        ear = BlinkDetector.eye_aspect_ratio(open_eye)
        assert ear > 0.25

    def test_eye_aspect_ratio_closed(self) -> None:
        """Test EAR for closed eye is below threshold."""
        closed_eye = _make_eye_landmarks(open_ratio=0.05)
        ear = BlinkDetector.eye_aspect_ratio(closed_eye)
        assert ear < 0.25

    def test_eye_aspect_ratio_zero_horizontal(self) -> None:
        """Test EAR returns 0 when horizontal distance is 0."""
        same_point = np.array([[0, 0]] * 6, dtype=np.float64)
        ear = BlinkDetector.eye_aspect_ratio(same_point)
        assert ear == 0.0

    def test_blink_detection_sequence(self) -> None:
        """Test blink is detected after open-close-open sequence."""
        detector = BlinkDetector(ear_threshold=0.25, consec_frames=2)

        open_eye = _make_eye_landmarks(open_ratio=0.3)
        closed_eye = _make_eye_landmarks(open_ratio=0.05)

        # Open frames
        for _ in range(3):
            _, blink = detector.process_frame(open_eye, open_eye)
            assert blink is False

        # Closed frames (enough for consec_frames)
        for _ in range(3):
            _, blink = detector.process_frame(closed_eye, closed_eye)
            assert blink is False  # Not yet, need open after close

        # Open again -> triggers blink
        _, blink = detector.process_frame(open_eye, open_eye)
        assert blink is True
        assert detector.get_blink_count() == 1

    def test_reset_clears_state(self) -> None:
        """Test reset clears all counters."""
        detector = BlinkDetector()
        detector.blink_counter = 5
        detector.frame_counter = 3
        detector.ear_history.append(0.2)

        detector.reset()

        assert detector.get_blink_count() == 0
        assert detector.frame_counter == 0
        assert len(detector.ear_history) == 0


class TestTextureAnalyzer:
    """Tests for TextureAnalyzer class."""

    def test_lbp_produces_valid_histogram(self) -> None:
        """Test LBP histogram has correct shape and sums to ~1."""
        analyzer = TextureAnalyzer()
        # Random image simulates varied texture
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        hist = analyzer.compute_lbp_histogram(image)

        assert hist.shape == (256,)
        assert hist.sum() == pytest.approx(1.0, abs=0.01)

    def test_high_entropy_image_passes(self) -> None:
        """Test that high-entropy (varied texture) image passes liveness."""
        analyzer = TextureAnalyzer(threshold=0.3)
        # Random noise has high entropy
        image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        score, is_live = analyzer.analyze(image)

        assert score > 0.3
        assert is_live is True

    def test_uniform_image_fails(self) -> None:
        """Test that uniform (low entropy) image fails liveness."""
        analyzer = TextureAnalyzer(threshold=0.5)
        # Solid color has zero entropy
        image = np.ones((60, 60), dtype=np.uint8) * 128
        score, is_live = analyzer.analyze(image)

        assert score < 0.5
        assert is_live is False

    def test_lbp_computation(self) -> None:
        """Test LBP computation produces valid output."""
        analyzer = TextureAnalyzer()
        image = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
        lbp = analyzer.compute_lbp(image)

        assert lbp.shape == (18, 18)  # 20 - 2*radius
        assert lbp.dtype == np.uint8

    def test_bgr_image_accepted(self) -> None:
        """Test that BGR (3-channel) image is accepted."""
        analyzer = TextureAnalyzer(threshold=0.3)
        image = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
        score, is_live = analyzer.analyze(image)

        assert 0 <= score <= 1.0


class TestLivenessDetector:
    """Tests for combined LivenessDetector."""

    def test_check_liveness_returns_dict(self) -> None:
        """Test check_liveness returns dict with all expected keys."""
        detector = LivenessDetector()
        image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        result = detector.check_liveness(image)

        expected_keys = {
            "texture_score",
            "texture_live",
            "blink_count",
            "blink_live",
            "overall_live",
            "confidence",
        }
        assert set(result.keys()) == expected_keys

    def test_texture_live_passes(self) -> None:
        """Test high-texture image passes overall liveness."""
        detector = LivenessDetector(texture_threshold=0.3)
        image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        result = detector.check_liveness(image)

        assert result["texture_live"] is True
        assert result["overall_live"] is True

    def test_blink_boosts_confidence(self) -> None:
        """Test that blinks increase confidence score."""
        detector = LivenessDetector(texture_threshold=0.3, required_blinks=1)

        # Manually set blink count
        detector.blink_detector.blink_counter = 1

        image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        result = detector.check_liveness(image)

        # Confidence should be boosted by blink
        assert result["blink_live"] is True
        assert result["confidence"] >= result["texture_score"]

    def test_with_eye_landmarks(self) -> None:
        """Test liveness check with eye landmarks provided."""
        detector = LivenessDetector()
        image = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
        left_eye = _make_eye_landmarks(0.3)
        right_eye = _make_eye_landmarks(0.3)

        result = detector.check_liveness(image, eye_landmarks=(left_eye, right_eye))

        assert "blink_count" in result

    def test_reset(self) -> None:
        """Test reset clears blink detector state."""
        detector = LivenessDetector()
        detector.blink_detector.blink_counter = 5

        detector.reset()

        assert detector.blink_detector.get_blink_count() == 0
