"""Lightweight visualizations for Outdoor-VLN JSONL samples."""

from __future__ import annotations

import textwrap
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from outdoor_vln_relabel.io_utils import ensure_dir


def _load_pil():
    """Import PIL lazily with a clear error message."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError(
            "PIL/Pillow is required for sample visualization. Install pillow or "
            "use the bundled workspace runtime that includes PIL."
        ) from exc
    return Image, ImageDraw, ImageFont


# Cache fonts so we don't search the filesystem on every call.
_font_cache = {}


def _get_font(size: int = 14):
    """Return a readable TrueType font, falling back to the bitmap default."""
    # Already cached this size.
    if size in _font_cache:
        return _font_cache[size]

    Image, _ImageDraw, ImageFont = _load_pil()
    font = None

    # Try common Linux TrueType fonts in rough order of preference.
    _TTF_CANDIDATES = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for ttf_path in _TTF_CANDIDATES:
        try:
            font = ImageFont.truetype(ttf_path, size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        # Last resort: PIL default bitmap font (tiny but always available).
        font = ImageFont.load_default()

    _font_cache[size] = font
    return font


def _trajectory_points(record: dict) -> List[Tuple[float, float]]:
    """Return trajectory_xy as ego-view float tuples for visualization only.

    This does not modify saved data. It only rotates the displayed trajectory so
    that the initial forward direction points upward in the sample visualization.
    """
    points = []
    for item in record.get("trajectory_xy") or []:
        if len(item) >= 2:
            points.append((float(item[0]), float(item[1])))

    if len(points) < 2:
        return points

    x0, y0 = points[0]
    centered = [(x - x0, y - y0) for x, y in points]

    dx, dy = 0.0, 0.0
    for x, y in centered[1:min(len(centered), 30)]:
        if abs(x) + abs(y) > 1e-6:
            dx, dy = x, y
            break

    if abs(dx) + abs(dy) < 1e-6:
        return centered

    # Rotate initial motion to +Y. _scale_points keeps the original plot box,
    # start/goal colors, labels, and card layout unchanged.
    theta = math.atan2(dy, dx)
    angle = math.pi / 2.0 - theta
    ca = math.cos(angle)
    sa = math.sin(angle)

    return [
        (ca * x - sa * y, sa * x + ca * y)
        for x, y in centered
    ]


def _scale_points(
    points: Sequence[Tuple[float, float]],
    width: int,
    height: int,
    padding: int,
) -> List[Tuple[int, int]]:
    """Scale XY trajectory points into image coordinates."""
    if not points:
        return []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    drawable_w = max(width - 2 * padding, 1)
    drawable_h = max(height - 2 * padding, 1)
    scale = min(drawable_w / span_x, drawable_h / span_y)
    used_w = span_x * scale
    used_h = span_y * scale
    offset_x = padding + (drawable_w - used_w) / 2
    offset_y = padding + (drawable_h - used_h) / 2
    scaled = []
    for x, y in points:
        px = int(round(offset_x + (x - min_x) * scale))
        py = int(round(height - (offset_y + (y - min_y) * scale)))
        scaled.append((px, py))
    return scaled


def _draw_wrapped_text(draw, xy: Tuple[int, int], text: str, width_chars: int, fill, font, line_spacing: int = 20):
    """Draw wrapped text and return the next y coordinate."""
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars) or [""]:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_spacing
    return y


def plot_trajectory(record: dict, output_path: str) -> None:
    """Plot a 2D trajectory with Start/Goal labels and sample metadata."""
    Image, ImageDraw, ImageFont = _load_pil()
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        ensure_dir(path.parent)

    inst_font = _get_font(15)
    meta_font = _get_font(11)
    label_font = _get_font(11)

    width, height = 900, 620
    plot_top = 150
    plot_h = 400
    image = Image.new("RGB", (width, height), color=(248, 249, 247))
    draw = ImageDraw.Draw(image)

    instruction = str(record.get("instruction", ""))
    y = _draw_wrapped_text(draw, (28, 22), f"Instruction: {instruction}", 90, (20, 30, 30), inst_font, line_spacing=22)
    meta = (
        f"terrain={record.get('terrain')} | motion={record.get('motion')} | "
        f"confidence={record.get('confidence')}"
    )
    draw.text((28, y + 6), meta, fill=(70, 80, 80), font=meta_font)

    plot_box = (50, plot_top, width - 50, plot_top + plot_h)
    draw.rectangle(plot_box, outline=(190, 196, 190), width=2)
    points = _trajectory_points(record)
    scaled = _scale_points(points, width - 100, plot_h, 34)
    scaled = [(x + 50, y + plot_top) for x, y in scaled]
    if len(scaled) >= 2:
        draw.line(scaled, fill=(36, 114, 178), width=4)
        for point in scaled[1:-1: max(1, len(scaled) // 16)]:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(36, 114, 178))
    if scaled:
        start = scaled[0]
        goal = scaled[-1]
        draw.ellipse((start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7), fill=(40, 150, 80))
        draw.text((start[0] + 10, start[1] - 10), "Start", fill=(20, 90, 45), font=label_font)
        draw.ellipse((goal[0] - 7, goal[1] - 7, goal[0] + 7, goal[1] + 7), fill=(210, 80, 70))
        draw.text((goal[0] + 10, goal[1] - 10), "Goal", fill=(150, 45, 45), font=label_font)
    else:
        draw.text((70, plot_top + 40), "No trajectory_xy points", fill=(150, 45, 45), font=label_font)

    image.save(path)


def _resize_to_height(image, height: int):
    """Resize an image to a target height while preserving aspect ratio."""
    if image.height == 0:
        return image
    width = max(1, int(round(image.width * (height / image.height))))
    return image.resize((width, height))


def _maybe_load_image(path: Optional[str], height: int):
    """Load an optional image path, returning None on missing files."""
    if not path:
        return None
    image_path = Path(path)
    if not image_path.is_file():
        return None
    Image, _, _ = _load_pil()
    try:
        return _resize_to_height(Image.open(image_path).convert("RGB"), height)
    except OSError:
        return None


def plot_trajectory_only(
    record: dict,
    width: int = 1300,
    height: int = 480,
    padding: int = 40,
    line_color: Tuple[int, int, int] = (36, 114, 178),
    start_color: Tuple[int, int, int] = (40, 150, 80),
    goal_color: Tuple[int, int, int] = (210, 80, 70),
) -> "Image.Image":
    """Render a clean trajectory plot with Start/Goal markers, no text overlay.

    Returns a PIL Image for embedding into larger card layouts.
    """
    Image, ImageDraw, _ImageFont = _load_pil()
    label_font = _get_font(10)

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    points = _trajectory_points(record)
    scaled = _scale_points(points, width, height, padding)
    if len(scaled) >= 2:
        draw.line(scaled, fill=line_color, width=3)
        step = max(1, len(scaled) // 16)
        for pt in scaled[1:-1:step]:
            draw.ellipse((pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2), fill=line_color)
    if scaled:
        start = scaled[0]
        goal = scaled[-1]
        r = 6
        draw.ellipse((start[0] - r, start[1] - r, start[0] + r, start[1] + r), fill=start_color)
        draw.text((start[0] + 10, start[1] - 8), "Start", fill=start_color, font=label_font)
        draw.ellipse((goal[0] - r, goal[1] - r, goal[0] + r, goal[1] + r), fill=goal_color)
        draw.text((goal[0] + 10, goal[1] - 8), "Goal", fill=goal_color, font=label_font)
    else:
        draw.text((padding, height // 2), "No trajectory_xy points", fill=goal_color, font=label_font)

    # Light border
    draw.rectangle((0, 0, width - 1, height - 1), outline=(210, 210, 210), width=1)

    return image


def _landmark_lines(record: dict) -> List[str]:
    """Format landmark dictionaries for display."""
    lines = []
    for landmark in record.get("landmarks") or []:
        if not isinstance(landmark, dict):
            continue
        lines.append(
            f"- {landmark.get('name')} "
            f"({landmark.get('role')}, {landmark.get('relation')}, "
            f"score={landmark.get('score')})"
        )
    return lines or ["- none"]


def make_sample_card(
    record: dict, image_paths: Optional[List[str]], output_path: str
) -> None:
    """Create a single PNG card with trajectory, instruction, and landmarks."""
    Image, ImageDraw, ImageFont = _load_pil()
    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        ensure_dir(path.parent)

    temp_plot = path.with_suffix(".trajectory.tmp.png")
    plot_trajectory(record, str(temp_plot))
    trajectory_image = Image.open(temp_plot).convert("RGB")
    try:
        temp_plot.unlink()
    except OSError:
        pass

    side_images = []
    for image_path in image_paths or []:
        loaded = _maybe_load_image(image_path, 180)
        if loaded is not None:
            side_images.append(loaded)

    inst_font = _get_font(15)
    meta_font = _get_font(11)
    section_font = _get_font(12)
    lm_font = _get_font(11)

    width = 1000
    image_strip_h = 210 if side_images else 0
    text_h = 220
    height = trajectory_image.height + image_strip_h + text_h
    card = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(card)

    trajectory_resized = trajectory_image.resize((width, trajectory_image.height))
    card.paste(trajectory_resized, (0, 0))
    y = trajectory_resized.height + 18

    if side_images:
        x = 24
        draw.text((x, y), "Frames", fill=(40, 40, 40), font=section_font)
        y += 20
        for loaded in side_images[:2]:
            card.paste(loaded, (x, y))
            x += loaded.width + 18
        y += 190

    y = _draw_wrapped_text(
        draw,
        (24, y),
        f"Instruction: {record.get('instruction', '')}",
        100,
        (20, 30, 30),
        inst_font,
        line_spacing=22,
    )
    draw.text(
        (24, y + 4),
        (
            f"terrain={record.get('terrain')} | motion={record.get('motion')} | "
            f"confidence={record.get('confidence')}"
        ),
        fill=(70, 80, 80),
        font=meta_font,
    )
    y += 26
    draw.text((24, y), "Landmarks:", fill=(40, 40, 40), font=section_font)
    y += 20
    for line in _landmark_lines(record)[:6]:
        draw.text((44, y), line, fill=(70, 70, 70), font=lm_font)
        y += 18

    card.save(path)
