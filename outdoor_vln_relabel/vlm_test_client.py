#!/usr/bin/env python3
"""End-to-end demo: read RELLIS frames + structured features → VLM → multi-level instruction.

Usage::

    python vlm_test_client.py

Requires ``ANTHROPIC_API_KEY`` set in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from outdoor_vln_relabel.language.vlm_client import ClaudeVLMClient
from outdoor_vln_relabel.language.vlm_prompts import build_multilevel_vln_prompt
from outdoor_vln_relabel.language.vlm_schemas import ExtractedLandmark, VLMInstruction

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("vlm_test_client")

# -- Paths -------------------------------------------------------------------------
_OUTPUTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "rellis_00000_run_official_map"
)
_PAIRS_PATH = os.path.join(_OUTPUTS_DIR, "pairs_v4_filtered.jsonl")
_FRAME_PATTERN = (
    "/home/song/Outdoor-VLN/data/Rellis_3D_pylon_camera_node/"
    "Rellis-3D/00000/pylon_camera_node/frame{idx:06d}-{ts}.jpg"
)


def _resolve_outputs_path() -> str:
    """Resolve the outputs directory robustly, falling back to absolute path."""
    abs_path = os.path.abspath(_OUTPUTS_DIR)
    if os.path.isdir(abs_path):
        return abs_path
    alt = "/home/song/Outdoor-VLN/third_party/outdoor_vln_relabel/outputs/rellis_00000_run_official_map"
    if os.path.isdir(alt):
        return alt
    raise FileNotFoundError(f"Cannot find outputs directory. Tried: {abs_path}, {alt}")


def load_pairs(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all instruction-path pairs from a JSONL file."""
    pairs: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def pick_demo_pair(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select a pair with person landmarks and sufficient frame range for a rich demo."""
    # Prefer pairs with person landmarks AND good frame count
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for p in pairs:
        frame_count = p.get("end_idx", 0) - p.get("start_idx", 0)
        has_person = any(
            lm.get("name") == "person" for lm in p.get("landmarks", [])
        )
        score = frame_count + (100 if has_person else 0)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def resolve_frames(
    start_idx: int,
    end_idx: int,
    start_frame_path: str,
    num_frames: int = 5,
) -> Tuple[List[str], List[int]]:
    """Resolve ``num_frames`` real camera frame paths spaced across the segment.

    Returns:
        Tuple of (image_paths, selected_frame_indices).
    """
    if start_frame_path and os.path.exists(start_frame_path):
        # Derive base directory and timestamp pattern from the start frame
        base_dir = os.path.dirname(start_frame_path)
        # Scan available frames in the range
        available: List[Tuple[int, str]] = []
        for idx in range(start_idx, end_idx + 1):
            # Try to find the frame by glob pattern
            pattern = os.path.join(base_dir, f"frame{idx:06d}-*.jpg")
            import glob
            matches = glob.glob(pattern)
            if matches:
                available.append((idx, matches[0]))

        if len(available) >= num_frames:
            # Evenly sample
            step = max(1, len(available) // num_frames)
            selected = [available[i] for i in range(0, len(available), step)][:num_frames]
            return [p for _, p in selected], [i for i, _ in selected]

    # Fallback: use the start frame path pattern directly
    if start_frame_path and os.path.exists(start_frame_path):
        base_dir = os.path.dirname(start_frame_path)
        indices = [
            start_idx,
            start_idx + (end_idx - start_idx) // 4,
            start_idx + (end_idx - start_idx) // 2,
            start_idx + 3 * (end_idx - start_idx) // 4,
            end_idx,
        ]
        paths: List[str] = []
        valid_indices: List[int] = []
        for idx in indices:
            # Search for any timestamp
            import glob
            pattern = os.path.join(base_dir, f"frame{idx:06d}-*.jpg")
            matches = glob.glob(pattern)
            if matches:
                paths.append(matches[0])
                valid_indices.append(idx)
        if len(paths) >= 3:
            return paths[:num_frames], valid_indices[:num_frames]

    return [], []


def sample_key_coords(trajectory_xy: List[List[float]], num_points: int = 5) -> List[Tuple[float, float]]:
    """Down-sample a dense trajectory to ``num_points`` key (x, y) points."""
    if len(trajectory_xy) <= num_points:
        return [tuple(pt) for pt in trajectory_xy]

    step = max(1, len(trajectory_xy) // num_points)
    key_pts: List[Tuple[float, float]] = []
    for i in range(0, len(trajectory_xy), step):
        x, y = trajectory_xy[i]
        key_pts.append((x, y))
        if len(key_pts) >= num_points:
            break
    return key_pts[:num_points]


def extract_semantics(landmarks: List[Dict[str, Any]]) -> List[str]:
    """Extract unique landmark names for prompt injection."""
    names: List[str] = []
    for lm in landmarks:
        name = lm.get("name", "")
        if name and name not in names:
            names.append(name)
    return names


def pretty_print_instruction(vlm_instruction: VLMInstruction) -> None:
    """Print the three-level instruction with formatting."""
    print("\n" + "=" * 70)
    print("  VLM 多层级导航指令生成结果")
    print("=" * 70)

    # Landmarks
    if vlm_instruction.extracted_landmarks:
        print("\n📌 提取的地标 (extracted_landmarks):")
        print("-" * 40)
        for lm in vlm_instruction.extracted_landmarks:
            print(f"  • {lm.name}  |  类别: {lm.category}  |  "
                  f"角色: {lm.role}  |  关系: {lm.relation}")
    else:
        print("\n📌 提取的地标: (无)")

    # Global
    print("\n🌍 宏观指令 (global_instruction):")
    print("-" * 40)
    print(textwrap.fill(
        vlm_instruction.global_instruction or "(未生成)",
        width=65, initial_indent="  ", subsequent_indent="  ",
    ))

    # Meso
    print("\n🏞️  中观指令 (meso_instruction):")
    print("-" * 40)
    print(textwrap.fill(
        vlm_instruction.meso_instruction or "(未生成)",
        width=65, initial_indent="  ", subsequent_indent="  ",
    ))

    # Micro
    print("\n🚶 微观指令 (micro_instruction):")
    print("-" * 40)
    print(textwrap.fill(
        vlm_instruction.micro_instruction or "(未生成)",
        width=65, initial_indent="  ", subsequent_indent="  ",
    ))

    print("\n" + "=" * 70)


def main() -> None:
    """Run the end-to-end VLM instruction generation demo."""
    print("🔧 Outdoor-VLN VLM 混合架构 — 指令生成测试")
    print("-" * 50)

    # 1. Load data
    outputs_dir = _resolve_outputs_path()
    pairs_path = os.path.join(outputs_dir, "pairs_v4_filtered.jsonl")
    print(f"📂 数据源: {pairs_path}")

    pairs = load_pairs(pairs_path)
    print(f"   已加载 {len(pairs)} 条 pair 记录")

    pair = pick_demo_pair(pairs)
    print(f"   选中 segment: {pair['segment_id']}  (instruction_id={pair['instruction_id']})")
    print(f"   原始指令: {pair['instruction']}")

    # 2. Extract structured features
    trajectory_xy = pair.get("trajectory_xy", [])
    terrain = pair.get("terrain", "grass")
    landmarks = pair.get("landmarks", [])

    key_coords = sample_key_coords(trajectory_xy, num_points=5)
    semantics = extract_semantics(landmarks)

    print(f"\n📊 注入特征:")
    print(f"   关键坐标 (5): {key_coords}")
    print(f"   地形: {terrain}")
    print(f"   语义地标: {semantics}")

    # 3. Resolve camera frames
    start_idx = pair.get("start_idx", 0)
    end_idx = pair.get("end_idx", start_idx + 100)
    start_frame = pair.get("start_frame", "")
    frame_paths, frame_indices = resolve_frames(start_idx, end_idx, start_frame, num_frames=5)
    print(f"\n📷 相机帧 ({len(frame_paths)} 张):")
    for i, (path, idx) in enumerate(zip(frame_paths, frame_indices)):
        print(f"   [{i+1}] frame {idx}: {os.path.basename(path)}")

    if not frame_paths:
        print("❌ 未找到真实相机帧，请检查数据路径。")
        sys.exit(1)

    # 4. Build prompt
    prompt = build_multilevel_vln_prompt(
        trajectory_coords=key_coords,
        terrain_types=[terrain] if terrain else None,
        detected_semantics=semantics if semantics else None,
        num_rgb_images=len(frame_paths),
        num_semantic_images=0,
        has_bev=False,
    )
    print(f"\n📝 生成的 Prompt (前300字):")
    print("-" * 40)
    print(prompt[:300] + "...")

    # 5. Call VLM
    print("\n🤖 调用 Claude VLM...")
    try:
        client = ClaudeVLMClient()
    except ValueError as exc:
        print(f"❌ {exc}")
        print("   请设置环境变量: export ANTHROPIC_API_KEY=sk-...")
        sys.exit(1)

    try:
        raw = client.generate(prompt, images=frame_paths, temperature=0.3, max_tokens=2048)
    except Exception as exc:
        print(f"❌ API 调用失败: {exc}")
        sys.exit(1)

    # 6. Parse and display
    print("\n📋 原始响应 (前500字):")
    print("-" * 40)
    print(raw[:500])

    parsed = _parse_json(raw)
    instruction = VLMInstruction.from_dict(parsed)

    if parsed.get("_error"):
        print(f"\n⚠️  JSON 解析警告: {parsed['_error']}")

    if instruction.global_instruction or instruction.micro_instruction:
        pretty_print_instruction(instruction)
    else:
        print("\n⚠️  未能从响应中解析出结构化指令，完整原始输出如下:")
        print(raw)


def _parse_json(raw: str) -> Dict[str, Any]:
    """Inline JSON parsing used in main for single call."""
    if not raw or not raw.strip():
        return {"_error": "empty_response"}
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_error": "json_parse_failed", "_raw": raw}


if __name__ == "__main__":
    main()
