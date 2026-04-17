from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from violation_model import ViolationModel, track_feature_dict
from vehicle_detector import VehicleDetector


Point = Tuple[float, float]


def open_video_source(source: Union[str, int]) -> cv2.VideoCapture:
    if isinstance(source, str) and source.lower().startswith(("http://", "https://", "rtsp://")):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def centroid(rect: Tuple[int, int, int, int]) -> Point:
    x, y, w, h = rect
    return x + w / 2.0, y + h / 2.0


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(v1: Point, v2: Point) -> float:
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-5 or mag2 < 1e-5:
        return 0.0
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    value = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(value))


@dataclass
class VehicleTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Point
    last_seen: int
    history: List[Point] = field(default_factory=list)
    speed_kmh: float = 0.0
    wrong_way: bool = False
    overspeed: bool = False
    illegal_uturn: bool = False
    seen_count: int = 1
    overspeed_votes: int = 0
    wrong_way_votes: int = 0
    uturn_votes: int = 0
    confidence_level: float = 0.0
    overspeed_confidence: float = 0.0
    wrong_way_confidence: float = 0.0
    uturn_confidence: float = 0.0
    detection_confidence: float = 0.0
    class_name: str = "vehicle"
    world_history: List[Point] = field(default_factory=list)
    counted: bool = False

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        center: Point,
        frame_index: int,
        detection_confidence: float = 0.0,
        class_name: str = "vehicle",
    ) -> None:
        self.bbox = bbox
        self.center = center
        self.last_seen = frame_index
        self.seen_count += 1
        self.detection_confidence = detection_confidence
        self.class_name = class_name
        self.history.append(center)
        if len(self.history) > 40:
            self.history = self.history[-40:]


class TrafficAnalyzer:
    def __init__(
        self,
        capture: cv2.VideoCapture,
        pixels_per_meter: float,
        speed_limit_kmh: float,
        expected_direction: str,
        uturn_angle_threshold: float = 130.0,
    ) -> None:
        self.capture = capture
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.speed_limit_kmh = speed_limit_kmh
        self.expected_direction = expected_direction
        self.uturn_angle_threshold = uturn_angle_threshold
        self.frame_index = 0
        self.next_track_id = 1
        self.tracks: Dict[int, VehicleTrack] = {}
        self.max_missing_frames = 20
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=28,
            detectShadows=True,
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_time = time.time()
        self.violation_model = ViolationModel()
        self.vehicle_detector = VehicleDetector()
        self.counted_ids: Set[int] = set()
        self.total_count = 0
        self.heatmap: Optional[np.ndarray] = None
        self.roi_polygon: Optional[np.ndarray] = None
        self.homography_matrix: Optional[np.ndarray] = None
        self.counting_line_y = 0
        self.speed_samples: List[float] = []
        self.last_metrics = {
            "vehicles": 0,
            "total_count": 0,
            "avg_speed_kmh": 0.0,
            "overspeeding": 0,
            "wrong_way": 0,
            "illegal_uturns": 0,
            "overall_confidence_pct": 0.0,
            "alert_confidence_pct": 0.0,
            "congestion_level": "Low",
            "frame": 0,
            "alerts": [],
            "finished": False,
        }

    def release(self) -> None:
        self.capture.release()

    def save_session_plot(self, output_dir: Path) -> Optional[Path]:
        if not self.speed_samples:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"speed_distribution_{int(time.time())}.png"
        plt.figure(figsize=(8, 4.5))
        plt.hist(self.speed_samples, bins=12, color="#00a6a6", edgecolor="#10202b")
        plt.title("Vehicle Speed Distribution")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Vehicle frequency")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        return plot_path

    def snapshot(self) -> dict:
        return self.last_metrics

    def next_jpeg(self) -> Tuple[bool, bytes]:
        ok, frame = self.capture.read()
        if not ok:
            self.last_metrics["finished"] = True
            return False, b""

        annotated = self.process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return False, b""
        return True, buffer.tobytes()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_index += 1
        now = time.time()
        dt = max(1 / 60, now - self.prev_time)
        self.prev_time = now

        frame = self.resize_frame(frame)
        self.ensure_scene_geometry(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracked_detections = self.detect_tracked_vehicles(frame)
        if tracked_detections:
            self.update_tracks_from_tracker(tracked_detections)
            self.update_world_speeds(dt)
        else:
            detections = self.detect_moving_vehicles(frame)
            self.update_tracks(detections)
            self.apply_optical_flow(gray, dt)
        self.update_counting_and_heatmap(frame)
        self.evaluate_events()
        annotated = self.draw_overlay(frame)
        self.prev_gray = gray
        return annotated

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= 960:
            return frame
        scale = 960.0 / width
        return cv2.resize(frame, (960, int(height * scale)), interpolation=cv2.INTER_AREA)

    def ensure_scene_geometry(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if self.heatmap is None or self.heatmap.shape[:2] != frame.shape[:2]:
            self.heatmap = np.zeros((height, width), dtype=np.float32)
        if self.roi_polygon is None or self.roi_polygon.shape[0] != 4:
            self.roi_polygon = np.array(
                [
                    (int(width * 0.14), int(height * 0.96)),
                    (int(width * 0.86), int(height * 0.96)),
                    (int(width * 0.62), int(height * 0.52)),
                    (int(width * 0.38), int(height * 0.52)),
                ],
                dtype=np.float32,
            )
            dst_points = np.array(
                [
                    (0.0, height * 0.60),
                    (width * 0.72, height * 0.60),
                    (width * 0.72, 0.0),
                    (0.0, 0.0),
                ],
                dtype=np.float32,
            )
            self.homography_matrix = cv2.getPerspectiveTransform(self.roi_polygon, dst_points)
            self.counting_line_y = int(height * 0.73)

    def detect_tracked_vehicles(self, frame: np.ndarray) -> List[dict]:
        tracked = self.vehicle_detector.track(frame)
        if not tracked:
            return []

        filtered = []
        for item in tracked:
            x, y, w, h = item["bbox"]
            bottom_center = (x + w / 2.0, y + h)
            if self.roi_polygon is not None:
                inside = cv2.pointPolygonTest(self.roi_polygon, bottom_center, False) >= 0
                if not inside:
                    continue
            filtered.append(item)
        return filtered

    def detect_moving_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detector_boxes = self.vehicle_detector.detect(frame)
        if detector_boxes:
            return detector_boxes

        mask = self.bg_subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Tuple[int, int, int, int]] = []
        frame_area = frame.shape[0] * frame.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1200 or area > frame_area * 0.35:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            aspect_ratio = w / max(1, h)
            if w < 36 or h < 24 or box_area < 1800:
                continue
            if aspect_ratio < 0.35 or aspect_ratio > 5.5:
                continue
            detections.append((x, y, w, h))
        return detections

    def update_tracks_from_tracker(self, tracked_detections: List[dict]) -> None:
        seen_ids = set()
        for item in tracked_detections:
            track_id = item["track_id"]
            bbox = item["bbox"]
            center = centroid(bbox)
            if track_id not in self.tracks:
                self.tracks[track_id] = VehicleTrack(
                    track_id=track_id,
                    bbox=bbox,
                    center=center,
                    last_seen=self.frame_index,
                    history=[center],
                    detection_confidence=item.get("confidence", 0.0),
                    class_name=item.get("class_name", "vehicle"),
                )
            else:
                self.tracks[track_id].update(
                    bbox,
                    center,
                    self.frame_index,
                    detection_confidence=item.get("confidence", 0.0),
                    class_name=item.get("class_name", "vehicle"),
                )
            seen_ids.add(track_id)

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track_id not in seen_ids and self.frame_index - track.last_seen > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def project_point(self, point: Point) -> Point:
        if self.homography_matrix is None:
            return point
        pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)[0][0]
        return float(transformed[0]), float(transformed[1])

    def update_world_speeds(self, dt: float) -> None:
        for track in self.tracks.values():
            x, y, w, h = track.bbox
            ground_point = (x + w / 2.0, y + h)
            projected = self.project_point(ground_point)
            track.world_history.append(projected)
            if len(track.world_history) > 40:
                track.world_history = track.world_history[-40:]

            if len(track.world_history) < 2:
                continue
            distance_px = distance(track.world_history[-1], track.world_history[-2])
            instant_speed = ((distance_px / self.pixels_per_meter) / dt) * 3.6
            track.speed_kmh = 0.74 * track.speed_kmh + 0.26 * instant_speed
            if track.speed_kmh > 1:
                self.speed_samples.append(track.speed_kmh)

    def update_counting_and_heatmap(self, frame: np.ndarray) -> None:
        if self.heatmap is None:
            return
        self.heatmap *= 0.985
        for track in self.confirmed_tracks():
            x, y, w, h = track.bbox
            ground_point = (int(x + w / 2.0), int(y + h))
            cv2.circle(self.heatmap, ground_point, 24, 1.8, -1)

            if track.track_id in self.counted_ids or len(track.history) < 2:
                continue
            prev_y = track.history[-2][1]
            current_y = track.history[-1][1]
            crossed_down = prev_y < self.counting_line_y <= current_y
            crossed_up = prev_y > self.counting_line_y >= current_y
            if crossed_down or crossed_up:
                self.counted_ids.add(track.track_id)
                self.total_count += 1

    def update_tracks(self, detections: List[Tuple[int, int, int, int]]) -> None:
        assigned_tracks = set()

        for bbox in detections:
            center = centroid(bbox)
            best_id = None
            best_distance = 9999.0

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                d = distance(center, track.center)
                gate = max(45.0, bbox[2] * 0.8, bbox[3] * 0.8)
                if d < gate and d < best_distance:
                    best_distance = d
                    best_id = track_id

            if best_id is None:
                track = VehicleTrack(
                    track_id=self.next_track_id,
                    bbox=bbox,
                    center=center,
                    last_seen=self.frame_index,
                    history=[center],
                )
                self.tracks[self.next_track_id] = track
                assigned_tracks.add(self.next_track_id)
                self.next_track_id += 1
            else:
                self.tracks[best_id].update(bbox, center, self.frame_index)
                assigned_tracks.add(best_id)

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if self.frame_index - track.last_seen > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def apply_optical_flow(self, gray: np.ndarray, dt: float) -> None:
        if self.prev_gray is None:
            return

        for track in self.tracks.values():
            x, y, w, h = track.bbox
            pad = 8
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            roi_prev = self.prev_gray[y1:y2, x1:x2]
            roi_next = gray[y1:y2, x1:x2]
            if roi_prev.size == 0 or roi_next.size == 0:
                continue

            points = cv2.goodFeaturesToTrack(
                roi_prev,
                maxCorners=60,
                qualityLevel=0.02,
                minDistance=6,
                blockSize=5,
            )
            if points is None:
                self.estimate_from_centroids(track, dt)
                continue

            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                roi_prev,
                roi_next,
                points,
                None,
                winSize=(17, 17),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            if new_points is None or status is None:
                self.estimate_from_centroids(track, dt)
                continue

            good_old = points[status.flatten() == 1]
            good_new = new_points[status.flatten() == 1]
            if len(good_old) < 4:
                self.estimate_from_centroids(track, dt)
                continue

            flow = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
            median_flow = np.median(flow, axis=0)
            px_per_second = float(np.linalg.norm(median_flow) / dt)
            instant_speed = (px_per_second / self.pixels_per_meter) * 3.6
            track.speed_kmh = 0.78 * track.speed_kmh + 0.22 * instant_speed

    def estimate_from_centroids(self, track: VehicleTrack, dt: float) -> None:
        if len(track.history) < 2:
            return
        px = distance(track.history[-1], track.history[-2])
        instant_speed = ((px / dt) / self.pixels_per_meter) * 3.6
        track.speed_kmh = 0.82 * track.speed_kmh + 0.18 * instant_speed

    def evaluate_events(self) -> None:
        alerts = []
        overspeeding = wrong_way = illegal_uturns = 0
        confidence_values = []
        alert_confidence_values = []

        confirmed = self.confirmed_tracks()
        for track in confirmed:
            direction_vector = self.direction_vector(track)
            raw_overspeed, raw_wrong_way, raw_uturn = self.predict_violations(track, direction_vector)
            confidence_values.append(track.confidence_level)

            track.overspeed_votes = self.update_votes(track.overspeed_votes, raw_overspeed)
            track.wrong_way_votes = self.update_votes(track.wrong_way_votes, raw_wrong_way)
            track.uturn_votes = self.update_votes(track.uturn_votes, raw_uturn)

            track.overspeed = track.overspeed_votes >= 4
            track.wrong_way = track.wrong_way_votes >= 5
            track.illegal_uturn = track.uturn_votes >= 5

            if track.overspeed:
                overspeeding += 1
                alert_confidence_values.append(track.overspeed_confidence)
                alerts.append(
                    f"Vehicle {track.track_id}: overspeeding ({track.speed_kmh:.1f} km/h, {track.overspeed_confidence * 100:.0f}% confidence)"
                )
            if track.wrong_way:
                wrong_way += 1
                alert_confidence_values.append(track.wrong_way_confidence)
                alerts.append(f"Vehicle {track.track_id}: wrong-way driving ({track.wrong_way_confidence * 100:.0f}% confidence)")
            if track.illegal_uturn:
                illegal_uturns += 1
                alert_confidence_values.append(track.uturn_confidence)
                alerts.append(f"Vehicle {track.track_id}: possible illegal U-turn ({track.uturn_confidence * 100:.0f}% confidence)")

        speeds = [track.speed_kmh for track in confirmed if track.speed_kmh > 1]
        avg_speed = round(float(np.mean(speeds)) if speeds else 0.0, 1)
        congestion = self.congestion_level(len(confirmed), avg_speed)
        self.last_metrics = {
            "vehicles": len(confirmed),
            "total_count": self.total_count,
            "avg_speed_kmh": avg_speed,
            "overspeeding": overspeeding,
            "wrong_way": wrong_way,
            "illegal_uturns": illegal_uturns,
            "overall_confidence_pct": round(float(np.mean(confidence_values) * 100) if confidence_values else 0.0, 1),
            "alert_confidence_pct": round(float(np.mean(alert_confidence_values) * 100) if alert_confidence_values else 0.0, 1),
            "congestion_level": congestion,
            "frame": self.frame_index,
            "alerts": alerts[-8:],
            "finished": False,
        }

    def confirmed_tracks(self) -> List[VehicleTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.seen_count >= 8 and self.track_area(track) >= 1800 and self.path_length(track) >= 12
        ]

    def update_votes(self, votes: int, condition: bool) -> int:
        if condition:
            return min(12, votes + 1)
        return max(0, votes - 2)

    def congestion_level(self, active_vehicles: int, avg_speed: float) -> str:
        if active_vehicles == 0:
            return "Low"
        if active_vehicles >= 14 or (active_vehicles >= 8 and avg_speed < 12):
            return "High"
        if active_vehicles >= 6 or avg_speed < 22:
            return "Moderate"
        return "Low"

    def predict_violations(self, track: VehicleTrack, direction_vector: Point) -> Tuple[bool, bool, bool]:
        features = track_feature_dict(track, self.speed_limit_kmh, self.expected_direction)
        probabilities = self.violation_model.predict(features)
        if probabilities is None:
            track.overspeed_confidence = self.rule_confidence(track.speed_kmh - self.speed_limit_kmh, 7.0)
            track.wrong_way_confidence = 0.84 if self.is_wrong_way(direction_vector) else 0.16
            track.uturn_confidence = 0.82 if self.is_uturn(track, direction_vector) else 0.14
            track.confidence_level = max(track.overspeed_confidence, track.wrong_way_confidence, track.uturn_confidence)
            return (
                track.speed_kmh > self.speed_limit_kmh,
                self.is_wrong_way(direction_vector),
                self.is_uturn(track, direction_vector),
            )

        track.overspeed_confidence = max(
            self.rule_confidence(track.speed_kmh - self.speed_limit_kmh, 7.0),
            probabilities.get("overspeeding", 0.0),
        )
        track.wrong_way_confidence = probabilities.get("wrong_way", 0.0)
        track.uturn_confidence = probabilities.get("illegal_uturn", 0.0)
        track.confidence_level = max(track.overspeed_confidence, track.wrong_way_confidence, track.uturn_confidence)
        return (
            track.overspeed_confidence >= 0.58,
            track.wrong_way_confidence >= 0.60,
            track.uturn_confidence >= 0.62,
        )

    def rule_confidence(self, value: float, scale: float) -> float:
        return float(1.0 / (1.0 + math.exp(-(value / max(1.0, scale)))))

    def track_area(self, track: VehicleTrack) -> int:
        return track.bbox[2] * track.bbox[3]

    def path_length(self, track: VehicleTrack) -> float:
        if len(track.history) < 2:
            return 0.0
        return sum(distance(a, b) for a, b in zip(track.history, track.history[1:]))

    def direction_vector(self, track: VehicleTrack) -> Point:
        if len(track.history) < 6:
            return 0.0, 0.0
        start = track.history[max(0, len(track.history) - 12)]
        end = track.history[-1]
        return end[0] - start[0], end[1] - start[1]

    def is_wrong_way(self, vector: Point) -> bool:
        dx, dy = vector
        if math.hypot(dx, dy) < 26:
            return False
        expected = self.expected_direction
        if expected == "right":
            return dx < -8
        if expected == "left":
            return dx > 8
        if expected == "down":
            return dy < -8
        if expected == "up":
            return dy > 8
        return False

    def is_uturn(self, track: VehicleTrack, current_vector: Point) -> bool:
        if len(track.history) < 25 or math.hypot(*current_vector) < 24 or self.path_length(track) < 70:
            return False

        early_start = track.history[0]
        early_end = track.history[min(8, len(track.history) - 1)]
        initial_vector = early_end[0] - early_start[0], early_end[1] - early_start[1]
        if math.hypot(*initial_vector) < 24:
            return False
        turn_angle = angle_between(initial_vector, current_vector)
        return turn_angle >= self.uturn_angle_threshold

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        self.draw_scene_guides(overlay)
        for track in self.confirmed_tracks():
            x, y, w, h = track.bbox
            status = "NORMAL"
            color = (56, 189, 248)
            if track.overspeed:
                status = "OVERSPEED"
                color = (45, 45, 230)
            elif track.wrong_way or track.illegal_uturn:
                status = "VIOLATION"
                color = (0, 176, 255)

            self.draw_vehicle_box(overlay, track, color, status)

            points = [(int(px), int(py)) for px, py in track.history[-18:]]
            for p1, p2 in zip(points, points[1:]):
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

        self.draw_panel(overlay)
        return overlay

    def draw_scene_guides(self, frame: np.ndarray) -> None:
        if self.roi_polygon is not None:
            polygon = self.roi_polygon.astype(int)
            guide = frame.copy()
            cv2.fillPoly(guide, [polygon], (12, 90, 90))
            cv2.addWeighted(guide, 0.08, frame, 0.92, 0, frame)
            cv2.polylines(frame, [polygon], True, (0, 190, 190), 2, cv2.LINE_AA)

        if self.heatmap is not None and float(np.max(self.heatmap)) > 0.05:
            heat = np.clip(self.heatmap / max(0.1, float(np.max(self.heatmap))), 0, 1)
            heat_u8 = np.uint8(heat * 255)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
            mask = heat_u8 > 12
            frame[mask] = cv2.addWeighted(frame, 0.72, heat_color, 0.28, 0)[mask]

        if self.counting_line_y:
            cv2.line(frame, (0, self.counting_line_y), (frame.shape[1], self.counting_line_y), (255, 214, 10), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "Counting line",
                (18, max(110, self.counting_line_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (255, 214, 10),
                2,
                cv2.LINE_AA,
            )

    def draw_vehicle_box(self, frame: np.ndarray, track: VehicleTrack, color: Tuple[int, int, int], status: str) -> None:
        x, y, w, h = track.bbox
        x2, y2 = x + w, y + h

        cv2.rectangle(frame, (x + 2, y + 2), (x2 + 2, y2 + 2), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 3, cv2.LINE_AA)

        corner = max(12, min(26, int(min(w, h) * 0.28)))
        for start, end in [
            ((x, y), (x + corner, y)),
            ((x, y), (x, y + corner)),
            ((x2, y), (x2 - corner, y)),
            ((x2, y), (x2, y + corner)),
            ((x, y2), (x + corner, y2)),
            ((x, y2), (x, y2 - corner)),
            ((x2, y2), (x2 - corner, y2)),
            ((x2, y2), (x2, y2 - corner)),
        ]:
            cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)

        label = f"ID {track.track_id}  {track.speed_kmh:.1f} km/h"
        label_w, label_h = self.text_size(label, 0.55, 2)
        confidence_text = f"{track.confidence_level * 100:.0f}% confidence"
        status_w, _ = self.text_size(status, 0.46, 1)
        confidence_w, _ = self.text_size(confidence_text, 0.42, 1)
        chip_w = max(label_w, status_w, confidence_w) + 22
        chip_h = 66
        chip_x = max(4, min(x, frame.shape[1] - chip_w - 4))
        chip_y = y - chip_h - 8 if y - chip_h - 8 > 4 else min(y2 + 8, frame.shape[0] - chip_h - 4)

        panel = frame.copy()
        cv2.rectangle(panel, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (13, 20, 28), -1)
        cv2.rectangle(panel, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), color, 2, cv2.LINE_AA)
        cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (chip_x, chip_y), (chip_x + 6, chip_y + chip_h), color, -1)
        cv2.putText(frame, label, (chip_x + 13, chip_y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 248, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, status, (chip_x + 13, chip_y + 41), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (chip_x + 13, chip_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (196, 208, 218), 1, cv2.LINE_AA)

    def text_size(self, text: str, scale: float, thickness: int) -> Tuple[int, int]:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        return size

    def draw_panel(self, frame: np.ndarray) -> None:
        panel_h = 92
        cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_h), (16, 24, 32), -1)
        cv2.putText(
            frame,
            "Vision-Based Speed Estimation & Traffic Analysis",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        metrics = self.last_metrics
        line = (
            f"Active: {metrics['vehicles']}   Count: {metrics['total_count']}   Avg speed: {metrics['avg_speed_kmh']} km/h   "
            f"Overspeed: {metrics['overspeeding']}   Congestion: {metrics['congestion_level']}   "
            f"Overall confidence: {metrics['overall_confidence_pct']}%"
        )
        cv2.putText(frame, line, (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (214, 233, 255), 2)
