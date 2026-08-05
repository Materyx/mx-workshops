import csv
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "Files" / "pendulums"
OUTPUT_DIR = BASE_DIR / "Output" / "pendulums"

VIDEOS = [
    (FILES_DIR / "pendulum_1.mp4", "pendulum_1_trajectory"),
    (FILES_DIR / "pendulum_2.mp4", "pendulum_2_trajectory"),
]

BG_FRAMES = 30
THRESHOLD = 25
MIN_AREA = 120
MAX_AREA = 12000
MIN_CIRCULARITY = 0.35
ROI_Y_MIN = 0.35
GAP_FRAMES = 18
PROGRESS_EVERY = 250
PLOT_WIDTH_RATIO = 1.0
PLOT_HEIGHT_RATIO = 1 / 3
PLOT_MARGIN = 20
PLOT_X_TICK_STEP = 10
PLOT_PAD_LEFT = 40
PLOT_PAD_BOTTOM = 35
PLOT_PAD_TOP = 15
PLOT_PAD_RIGHT = 20
PLOT_CURVE_THICKNESS = 2
PIVOT_RADIUS = 8
PIVOT_COLOR = (0, 0, 0)

TrackPoint = tuple[float, int, int]


def jet_color_bgr(t: float) -> tuple[int, int, int]:
    t = float(np.clip(1.0 - t, 0.0, 1.0))
    red = np.clip(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


def normalized_time(timestamp: float, video_duration: float) -> float:
    if video_duration <= 0:
        return 0.0
    return float(np.clip(timestamp / video_duration, 0.0, 1.0))


def draw_segment_rainbow(
    frame: np.ndarray,
    segment: list[TrackPoint],
    video_duration: float,
) -> None:
    if not segment:
        return

    if len(segment) == 1:
        color = jet_color_bgr(normalized_time(segment[0][0], video_duration))
        cv2.circle(frame, (segment[0][1], segment[0][2]), 4, color, -1, cv2.LINE_AA)
        return

    for index in range(len(segment) - 1):
        t0 = normalized_time(segment[index][0], video_duration)
        t1 = normalized_time(segment[index + 1][0], video_duration)
        cv2.line(
            frame,
            (segment[index][1], segment[index][2]),
            (segment[index + 1][1], segment[index + 1][2]),
            jet_color_bgr((t0 + t1) / 2.0),
            2,
            cv2.LINE_AA,
        )

    last = segment[-1]
    cv2.circle(
        frame,
        (last[1], last[2]),
        4,
        jet_color_bgr(normalized_time(last[0], video_duration)),
        -1,
        cv2.LINE_AA,
    )


def draw_trajectory(
    frame: np.ndarray,
    segments: list[list[TrackPoint]],
    video_duration: float,
) -> None:
    for segment in segments:
        draw_segment_rainbow(frame, segment, video_duration)


def longest_segment(segments: list[list[TrackPoint]]) -> list[TrackPoint]:
    if not segments:
        return []
    return max(segments, key=len)


def pivot_from_circle_fit(points: list[tuple[int, int]]) -> tuple[int, int]:
    xs = np.array([point[0] for point in points], dtype=np.float64)
    ys = np.array([point[1] for point in points], dtype=np.float64)
    design = np.column_stack([2.0 * xs, 2.0 * ys, np.ones_like(xs)])
    rhs = xs * xs + ys * ys
    center_x, center_y, _offset = np.linalg.lstsq(design, rhs, rcond=None)[0]
    return int(round(center_x)), int(round(center_y))


def estimate_pivot(segments: list[list[TrackPoint]]) -> tuple[int, int] | None:
    segment = longest_segment(segments)
    if len(segment) < 3:
        return None

    points = [(point[1], point[2]) for point in segment]
    return pivot_from_circle_fit(points)


def draw_pivot(frame: np.ndarray, pivot: tuple[int, int]) -> None:
    cv2.circle(frame, pivot, PIVOT_RADIUS, PIVOT_COLOR, -1, cv2.LINE_AA)


def flatten_points(
    segments: list[list[TrackPoint]],
    current: list[TrackPoint],
) -> list[TrackPoint]:
    points: list[TrackPoint] = []
    for segment in segments:
        points.extend(segment)
    points.extend(current)
    return points


def compute_theta(
    points: list[TrackPoint],
    pivot: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pivot_x, pivot_y = pivot
    times = np.array([point[0] for point in points], dtype=np.float32)
    thetas = np.array(
        [
            np.arctan2(point[1] - pivot_x, point[2] - pivot_y)
            for point in points
        ],
        dtype=np.float32,
    )
    return times, thetas


def theta_limits(points: list[TrackPoint], pivot: tuple[int, int]) -> float:
    _, thetas = compute_theta(points, pivot)
    peak = float(np.max(np.abs(thetas))) if len(thetas) else 0.0
    return max(peak * 1.1, 0.01)


def write_trajectory_csv(
    output_path: Path,
    points: list[TrackPoint],
    pivot: tuple[int, int],
) -> None:
    pivot_x, pivot_y = pivot
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["t", "x", "y", "theta"])
        for timestamp, x_coord, y_coord in points:
            theta = np.arctan2(x_coord - pivot_x, y_coord - pivot_y)
            writer.writerow([f"{timestamp:.4f}", x_coord, y_coord, f"{theta:.4f}"])


def plot_rect(
    frame_width: int,
    frame_height: int,
    full_frame: bool = False,
) -> tuple[int, int, int, int]:
    plot_w = int(frame_width * PLOT_WIDTH_RATIO) - 2 * PLOT_MARGIN
    if full_frame:
        plot_h = frame_height - 2 * PLOT_MARGIN
        x0 = PLOT_MARGIN
        y0 = PLOT_MARGIN
    else:
        plot_h = int(frame_height * PLOT_HEIGHT_RATIO)
        x0 = PLOT_MARGIN
        y0 = frame_height - plot_h - PLOT_MARGIN
    return x0, y0, plot_w, plot_h


def x_tick_values(video_duration: float) -> list[int]:
    if video_duration <= 0:
        return [0]
    return list(range(0, int(video_duration) + 1, PLOT_X_TICK_STEP))


def draw_theta_plot(
    frame: np.ndarray,
    times: np.ndarray,
    theta: np.ndarray,
    video_duration: float,
    current_time: float,
    y_limit: float | None = None,
    full_frame: bool = False,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    x0, y0, plot_w, plot_h = plot_rect(frame_width, frame_height, full_frame=full_frame)
    axis_color = (0, 0, 0)
    text_color = (0, 0, 0)
    zero_line_color = (80, 80, 80)

    left = x0 + PLOT_PAD_LEFT
    right = x0 + plot_w - PLOT_PAD_RIGHT
    top = y0 + PLOT_PAD_TOP
    bottom = y0 + plot_h - PLOT_PAD_BOTTOM
    center_y = (top + bottom) // 2
    half_height = max((bottom - top) // 2, 1)

    cv2.line(frame, (left, bottom), (right, bottom), axis_color, 1, cv2.LINE_AA)
    cv2.line(frame, (left, top), (left, bottom), axis_color, 1, cv2.LINE_AA)
    cv2.line(frame, (left, center_y), (right, center_y), zero_line_color, 1, cv2.LINE_AA)

    y_limit = y_limit if y_limit is not None else (
        max(float(np.max(np.abs(theta))) * 1.1, 0.01) if len(theta) else 0.01
    )
    duration = max(video_duration, 1e-6)

    visible = times <= current_time + 1e-6
    plot_times = times[visible]
    plot_values = theta[visible]

    if len(plot_times) >= 2:
        curve_points = []
        for timestamp, value in zip(plot_times, plot_values):
            px = int(left + (timestamp / duration) * (right - left))
            py = int(center_y - (value / y_limit) * half_height)
            curve_points.append((timestamp, px, py))

        for index in range(len(curve_points) - 1):
            t0 = normalized_time(curve_points[index][0], video_duration)
            t1 = normalized_time(curve_points[index + 1][0], video_duration)
            cv2.line(
                frame,
                (curve_points[index][1], curve_points[index][2]),
                (curve_points[index + 1][1], curve_points[index + 1][2]),
                jet_color_bgr((t0 + t1) / 2.0),
                PLOT_CURVE_THICKNESS,
                cv2.LINE_AA,
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if full_frame else 0.45
    thickness = 1

    for tick in x_tick_values(video_duration):
        tick_x = int(left + (tick / duration) * (right - left))
        cv2.line(frame, (tick_x, bottom), (tick_x, bottom + 5), axis_color, 1, cv2.LINE_AA)
        tick_label = str(tick)
        label_size = cv2.getTextSize(tick_label, font, font_scale, thickness)[0]
        cv2.putText(
            frame,
            tick_label,
            (tick_x - label_size[0] // 2, bottom + 22),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    cv2.putText(frame, "0", (x0 + 8, center_y + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
    if len(theta):
        peak = y_limit / 1.1
        if peak > 0:
            peak_label = f"{peak:.2f}"
            peak_size = cv2.getTextSize(peak_label, font, font_scale, thickness)[0]
            cv2.putText(
                frame,
                peak_label,
                (x0 + 8, top + peak_size[1] + 2),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"-{peak_label}",
                (x0 + 8, bottom - 4),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    cv2.putText(
        frame,
        "θ",
        (left - 18, top - 8),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def build_background(capture: cv2.VideoCapture, frame_count: int) -> np.ndarray:
    frames = []
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if not frames:
        raise RuntimeError("Не удалось прочитать кадры для построения фона")

    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def detect_bob(gray_frame: np.ndarray, background: np.ndarray, frame_height: int) -> tuple[int, int] | None:
    diff = cv2.absdiff(gray_frame, background)
    _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    roi_y = int(frame_height * ROI_Y_MIN)
    mask[:roi_y, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        lower_bias = cy / frame_height
        score = circularity + lower_bias
        if score > best_score:
            best_score = score
            best = (cx, cy)

    return best


def track_segments(
    capture: cv2.VideoCapture,
    background: np.ndarray,
    height: int,
    fps: float,
) -> tuple[list[list[TrackPoint]], int]:
    segments: list[list[TrackPoint]] = []
    current: list[TrackPoint] = []
    missed_frames = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        point = detect_bob(gray, background, height)

        if point is not None:
            track_point = (frame_index / fps, point[0], point[1])
            if missed_frames >= GAP_FRAMES and current:
                segments.append(current)
                current = [track_point]
            else:
                current.append(track_point)
            missed_frames = 0
        else:
            missed_frames += 1

        frame_index += 1

    if current:
        segments.append(current)

    return segments, frame_index


def process_pendulum(input_video: Path, output_stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / f"{output_stem}.mp4"
    output_image = OUTPUT_DIR / f"{output_stem}.png"
    output_csv = OUTPUT_DIR / f"{output_stem}.csv"
    base_stem = output_stem.removesuffix("_trajectory")
    output_projection = OUTPUT_DIR / f"{base_stem}_projection.png"

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else 0.0

    background = build_background(capture, BG_FRAMES)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    segments, total_frames_read = track_segments(capture, background, height, fps)
    all_points = flatten_points(segments, [])
    if not all_points:
        raise RuntimeError(f"Не удалось отследить груз маятника: {input_video}")

    pivot = estimate_pivot(segments)
    if pivot is None:
        raise RuntimeError(f"Не удалось оценить pivot маятника: {input_video}")

    y_limit = theta_limits(all_points, pivot)
    video_duration = max(video_duration, total_frames_read / fps)
    write_trajectory_csv(output_csv, all_points, pivot)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {output_video}")

    segments_render: list[list[TrackPoint]] = []
    current: list[TrackPoint] = []
    missed_frames = 0
    frame_index = 0
    point_cursor = 0
    last_frame = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        point = detect_bob(gray, background, height)

        if point is not None:
            track_point = (frame_index / fps, point[0], point[1])
            if missed_frames >= GAP_FRAMES and current:
                segments_render.append(current)
                current = [track_point]
            else:
                current.append(track_point)
            missed_frames = 0
            point_cursor += 1
        else:
            missed_frames += 1

        annotated = frame.copy()
        visible_segments = segments_render + ([current] if current else [])
        draw_trajectory(annotated, visible_segments, video_duration)

        current_time = frame_index / fps
        visible_points = all_points[:point_cursor]
        if len(visible_points) >= 2:
            times, plot_theta = compute_theta(visible_points, pivot)
            draw_theta_plot(
                annotated,
                times,
                plot_theta,
                video_duration,
                current_time,
                y_limit=y_limit,
            )

        draw_pivot(annotated, pivot)

        writer.write(annotated)
        last_frame = annotated
        frame_index += 1

        if frame_index % PROGRESS_EVERY == 0:
            print(f"{input_video.name}: кадр {frame_index}, точек {point_cursor}")

    capture.release()
    writer.release()

    if last_frame is None:
        raise RuntimeError(f"Видео не содержит кадров: {input_video}")

    summary = np.full((height, width, 3), 255, dtype=np.uint8)
    times, plot_theta = compute_theta(all_points, pivot)
    draw_theta_plot(
        summary,
        times,
        plot_theta,
        video_duration,
        video_duration,
        y_limit=y_limit,
        full_frame=True,
    )
    cv2.imwrite(str(output_image), summary)

    projection = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_trajectory(projection, segments, video_duration)
    draw_pivot(projection, pivot)
    cv2.imwrite(str(output_projection), projection)

    point_count = len(all_points)
    print(f"Готово: {input_video.name}, точек {point_count}")
    print(f"Видео: {output_video}")
    print(f"Изображение: {output_image}")
    print(f"CSV: {output_csv}")
    print(f"Проекция: {output_projection}")


def main() -> None:
    for input_video, output_stem in VIDEOS:
        print(f"Обработка {input_video.name}...")
        process_pendulum(input_video, output_stem)


if __name__ == "__main__":
    main()
