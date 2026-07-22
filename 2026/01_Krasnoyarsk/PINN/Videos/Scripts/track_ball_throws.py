import csv
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "Files" / "ball_throws"
OUTPUT_DIR = BASE_DIR / "Output" / "ball_throws"

INPUT_VIDEO = FILES_DIR / "ball_throws.mp4"
OUTPUT_STEM = "ball_throws_trajectories"

BG_FRAMES = 30
BG_THRESHOLD = 25
FRAME_DIFF_THRESHOLD = 5
RED_DOMINANCE_THRESHOLD = 0
START_RED_DOMINANCE_THRESHOLD = -5
START_MIN_RED = 100
START_MIN_X = 80
MIN_AREA = 50
MAX_AREA = 8000
MIN_CIRCULARITY = 0.25
GAP_FRAMES = 5
MIN_TRAJECTORY_POINTS = 5
X_BACK_THRESHOLD = 30
MAX_FORWARD_JUMP = 220
PROGRESS_EVERY = 25
TRAJECTORY_THICKNESS = 4
NOISE_SEED = 2
NOISE_LEVEL = 0.07
TAB10_BGR = [
    (180, 119, 31),   #1f77b4
    (14, 127, 255),   #ff7f0e
    (44, 160, 44),    #2ca02c
    (40, 39, 214),    #d62728
    (189, 103, 148),  #9467bd
    (207, 190, 23),   #17becf
    (194, 119, 227),  #e377c2
    (51, 51, 51),     #333333
    (34, 189, 188),   #bcbd22
    (75, 86, 140),    #8c564b
]
TRAJECTORY_COLOR_BOOST = 1.25
MEASUREMENT_POINT_RADIUS = 8


TrackPoint = tuple[float, int, int]


def tab10_color_bgr(index: int) -> tuple[int, int, int]:
    palette_index = index % len(TAB10_BGR)
    blue, green, red = TAB10_BGR[palette_index]
    factor = TRAJECTORY_COLOR_BOOST if palette_index != 7 else 1.0
    return (
        min(255, int(blue * factor)),
        min(255, int(green * factor)),
        min(255, int(red * factor)),
    )


def output_paths(label: str) -> dict[str, Path]:
    return {
        "video": OUTPUT_DIR / f"{OUTPUT_STEM}-{label}.mp4",
        "image": OUTPUT_DIR / f"{OUTPUT_STEM}-{label}.png",
        "data": OUTPUT_DIR / f"{OUTPUT_STEM}-{label}.csv",
    }


def apply_y_noise(trajectories: list[list[TrackPoint]]) -> list[list[TrackPoint]]:
    rng = np.random.default_rng(NOISE_SEED)
    all_y = [point[2] for track in trajectories for point in track]
    y_scale = float(np.std(all_y)) if all_y else 1.0
    noisy: list[list[TrackPoint]] = []
    for track in trajectories:
        noisy_track: list[TrackPoint] = []
        for timestamp, x_coord, y_coord in track:
            noisy_y = y_coord + NOISE_LEVEL * y_scale * rng.standard_normal()
            noisy_track.append((timestamp, x_coord, int(round(noisy_y))))
        noisy.append(noisy_track)
    return noisy


def visible_trajectories_at_time(
    trajectories: list[list[TrackPoint]],
    current_time: float,
) -> list[list[TrackPoint]]:
    visible: list[list[TrackPoint]] = []
    for track in trajectories:
        points = [point for point in track if point[0] <= current_time + 1e-6]
        if points:
            visible.append(points)
    return visible


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


def red_dominance(frame_bgr: np.ndarray, x: int, y: int) -> int:
    blue, green, red = (int(frame_bgr[y, x, channel]) for channel in range(3))
    return red - max(blue, green)


def motion_mask(
    gray_frame: np.ndarray,
    background: np.ndarray,
    prev_gray: np.ndarray | None,
) -> np.ndarray:
    bg_diff = cv2.absdiff(gray_frame, background)
    _, bg_motion = cv2.threshold(bg_diff, BG_THRESHOLD, 255, cv2.THRESH_BINARY)

    if prev_gray is None:
        return bg_motion

    frame_diff = cv2.absdiff(gray_frame, prev_gray)
    _, frame_motion = cv2.threshold(frame_diff, FRAME_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_or(bg_motion, frame_motion)


def morph_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def pick_point(
    mask: np.ndarray,
    frame_bgr: np.ndarray,
    prev_point: tuple[int, int] | None,
    start_mode: bool = False,
) -> tuple[tuple[int, int] | None, bool]:
    contours, _ = cv2.findContours(morph_mask(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int], int, bool]] = []
    red_threshold = START_RED_DOMINANCE_THRESHOLD if start_mode else RED_DOMINANCE_THRESHOLD

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

        point = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        if point[0] < START_MIN_X:
            continue
        redness = red_dominance(frame_bgr, point[0], point[1])
        red_value = int(frame_bgr[point[1], point[0], 2])
        min_red = red_threshold
        if not start_mode and prev_point is not None:
            dx_hint = point[0] - prev_point[0]
            if dx_hint < -X_BACK_THRESHOLD:
                min_red = START_RED_DOMINANCE_THRESHOLD
            elif 0 < dx_hint <= MAX_FORWARD_JUMP:
                min_red = START_RED_DOMINANCE_THRESHOLD

        if start_mode:
            if redness < min_red or red_value < START_MIN_RED:
                continue
            score = -float(point[0]) + redness + circularity * 5.0
            candidates.append((score, point, redness, redness < 4))
            continue

        if redness < min_red:
            continue

        if prev_point is not None:
            dx = point[0] - prev_point[0]
            if dx > MAX_FORWARD_JUMP:
                continue
            distance = float(np.hypot(dx, point[1] - prev_point[1]))
            score = redness + circularity * 10.0 - distance * 0.01
        else:
            score = redness + circularity * 10.0

        candidates.append((score, point, redness, redness < 4))

    if not candidates:
        return None, False

    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, best, _redness, bg_like = candidates[0]
    return best, bg_like


def detect_ball(
    frame_bgr: np.ndarray,
    gray_frame: np.ndarray,
    background: np.ndarray,
    prev_gray: np.ndarray | None,
    prev_point: tuple[int, int] | None,
    start_mode: bool = False,
) -> tuple[tuple[int, int] | None, bool]:
    mask = motion_mask(gray_frame, background, prev_gray)
    point, bg_like = pick_point(mask, frame_bgr, prev_point, start_mode=start_mode)
    bg_detected = point is not None and not bg_like
    if point is None and prev_gray is not None:
        frame_only = cv2.threshold(
            cv2.absdiff(gray_frame, prev_gray),
            FRAME_DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )[1]
        point, bg_like = pick_point(frame_only, frame_bgr, prev_point, start_mode=start_mode)
        bg_detected = False
    return point, bg_detected


def should_start_new_throw(
    point: tuple[int, int],
    current_points: list[tuple[int, int]],
    missed_bg_frames: int,
) -> tuple[bool, str | None]:
    if not current_points:
        return False, None

    if point[0] < current_points[-1][0] - X_BACK_THRESHOLD:
        return True, "x_back"

    if missed_bg_frames >= GAP_FRAMES and point[0] < current_points[-1][0]:
        return True, "gap"

    return False, None


def finalize_trajectory(
    finished: list[list[TrackPoint]],
    current_points: list[TrackPoint],
) -> None:
    if len(current_points) >= MIN_TRAJECTORY_POINTS:
        finished.append(current_points)


def draw_trajectory_tab10(
    frame: np.ndarray,
    points: list[TrackPoint],
    trajectory_index: int,
) -> None:
    if not points:
        return

    color = tab10_color_bgr(trajectory_index)

    if len(points) >= 2:
        for index in range(len(points) - 1):
            cv2.line(
                frame,
                (points[index][1], points[index][2]),
                (points[index + 1][1], points[index + 1][2]),
                color,
                TRAJECTORY_THICKNESS,
                cv2.LINE_AA,
            )

    for _timestamp, x_coord, y_coord in points:
        cv2.circle(
            frame,
            (x_coord, y_coord),
            MEASUREMENT_POINT_RADIUS,
            color,
            -1,
            cv2.LINE_AA,
        )


def draw_trajectories(
    frame: np.ndarray,
    trajectories: list[list[TrackPoint]],
) -> None:
    for index, points in enumerate(trajectories):
        draw_trajectory_tab10(frame, points, index)


def write_trajectories_csv(
    output_path: Path,
    trajectories: list[list[TrackPoint]],
) -> None:
    if not trajectories:
        output_path.write_text("pid,t,x,y\n", encoding="utf-8")
        return

    max_len = max(len(track) for track in trajectories)
    header: list[str] = []
    for _ in trajectories:
        header.extend(["pid", "t", "x", "y"])

    rows: list[list[str | int | float]] = []
    for point_index in range(max_len):
        row: list[str | int | float] = []
        for track in trajectories:
            if point_index < len(track):
                timestamp, x_coord, y_coord = track[point_index]
                row.extend([point_index, f"{timestamp:.4f}", x_coord, y_coord])
            else:
                row.extend(["", "", "", ""])
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def render_trajectory_video(
    trajectories: list[list[TrackPoint]],
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> None:
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {INPUT_VIDEO}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {output_path}")

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        current_time = frame_index / fps
        annotated = frame.copy()
        draw_trajectories(
            annotated,
            visible_trajectories_at_time(trajectories, current_time),
        )
        writer.write(annotated)
        frame_index += 1

    capture.release()
    writer.release()


def save_trajectory_image(
    trajectories: list[list[TrackPoint]],
    output_path: Path,
    width: int,
    height: int,
) -> None:
    summary = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_trajectories(summary, trajectories)
    cv2.imwrite(str(output_path), summary)


def export_measurement_set(
    label: str,
    trajectories: list[list[TrackPoint]],
    width: int,
    height: int,
    fps: float,
) -> dict[str, Path]:
    paths = output_paths(label)
    render_trajectory_video(trajectories, paths["video"], width, height, fps)
    save_trajectory_image(trajectories, paths["image"], width, height)
    write_trajectories_csv(paths["data"], trajectories)
    return paths


def process_ball_throws() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    good_paths = output_paths("good")

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {INPUT_VIDEO}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if total_frames > 0 else 0.0

    background = build_background(capture, BG_FRAMES)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(
        str(good_paths["video"]),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {good_paths['video']}")

    finished: list[list[TrackPoint]] = []
    current_points: list[TrackPoint] = []
    missed_bg_frames = 0
    frame_index = 0
    last_frame = None
    prev_gray: np.ndarray | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_point = (current_points[-1][1], current_points[-1][2]) if current_points else None
        between_throws = not current_points or missed_bg_frames >= GAP_FRAMES
        point, _ = detect_ball(
            frame,
            gray,
            background,
            prev_gray,
            None if between_throws else prev_point,
            start_mode=between_throws,
        )

        if point is not None:
            start_new, _ = should_start_new_throw(
                point,
                [(track_point[1], track_point[2]) for track_point in current_points],
                missed_bg_frames,
            )

            track_point = (frame_index / fps, point[0], point[1])
            if start_new:
                finalize_trajectory(finished, current_points)
                current_points = [track_point]
            else:
                current_points.append(track_point)

            missed_bg_frames = 0
        else:
            missed_bg_frames += 1

        visible = finished.copy()
        if current_points:
            visible.append(current_points)

        annotated = frame.copy()
        draw_trajectories(annotated, visible)
        writer.write(annotated)
        last_frame = annotated
        prev_gray = gray
        frame_index += 1

        if frame_index % PROGRESS_EVERY == 0:
            point_count = sum(len(points) for points in visible)
            print(
                f"Кадр {frame_index}, точек: {point_count}, "
                f"траекторий: {len(finished) + (1 if current_points else 0)}"
            )

    finalize_trajectory(finished, current_points)
    video_duration = max(video_duration, frame_index / fps)

    capture.release()
    writer.release()

    if last_frame is None:
        raise RuntimeError("Видео не содержит кадров")

    good_trajectories = finished.copy()
    save_trajectory_image(good_trajectories, good_paths["image"], width, height)
    write_trajectories_csv(good_paths["data"], good_trajectories)

    bad_trajectories = apply_y_noise(good_trajectories)
    bad_paths = export_measurement_set("bad", bad_trajectories, width, height, fps)

    point_count = sum(len(points) for points in good_trajectories)
    print(f"Готово: {len(good_trajectories)} траекторий, {point_count} точек")
    for index, points in enumerate(good_trajectories, start=1):
        xs = [point[1] for point in points]
        ys = [point[2] for point in points]
        print(
            f"  {index}: {len(points)} точек, "
            f"x=[{min(xs)}, {max(xs)}], y=[{min(ys)}, {max(ys)}]"
        )
    print(f"Good данные: {good_paths['data']}")
    print(f"Good видео: {good_paths['video']}")
    print(f"Good изображение: {good_paths['image']}")
    print(f"Bad данные: {bad_paths['data']}")
    print(f"Bad видео: {bad_paths['video']}")
    print(f"Bad изображение: {bad_paths['image']}")


if __name__ == "__main__":
    process_ball_throws()
