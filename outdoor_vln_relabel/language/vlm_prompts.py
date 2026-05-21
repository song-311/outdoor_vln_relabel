"""Dynamic prompt templates for multi-level VLM navigation instruction generation."""

from __future__ import annotations

from typing import List, Optional, Tuple


def build_multilevel_vln_prompt(
    trajectory_coords: List[Tuple[float, float]],
    terrain_types: Optional[List[str]] = None,
    detected_semantics: Optional[List[str]] = None,
    num_rgb_images: int = 5,
    num_semantic_images: int = 4,
    has_bev: bool = True,
) -> str:
    """Build a dynamic multi-level VLN prompt with injected structured features.

    Args:
        trajectory_coords: List of (x, y) coordinates for key path points.
        terrain_types: Optional terrain labels (e.g. ["grass", "mud_water"]).
        detected_semantics: Optional semantic landmark names from mask parsing.
        num_rgb_images: Number of RGB images attached (figures 1..N).
        num_semantic_images: Number of semantic mask images (figures N+1..).
        has_bev: Whether a BEV point-cloud image is included.

    Returns:
        Formatted prompt string ready for the VLM.
    """

    # -- Build image reference sentences -------------------------------------------------
    rgb_range = (
        f"图1到图{num_rgb_images}"
        if num_rgb_images > 1
        else f"图{num_rgb_images}"
    )
    rgb_desc = f"{rgb_range}为{num_rgb_images}个连续路径点的真实图像"

    image_parts = [rgb_desc]

    if num_semantic_images > 0:
        sem_start = num_rgb_images + 1
        sem_end = num_rgb_images + num_semantic_images
        sem_desc = (
            f"图{sem_start}到图{sem_end}为前{num_semantic_images}个路径点的语义图"
        )
        image_parts.append(sem_desc)

    if has_bev:
        bev_idx = num_rgb_images + num_semantic_images + 1
        image_parts.append(f"第{bev_idx}张为第三个路径点位置的点云生成的BEV图")

    image_context = "，".join(image_parts) + "。"

    # -- Landmark library ----------------------------------------------------------------
    terrain_defaults = ["缓坡上行", "起伏丘陵", "坡折点", "地势开阔"]
    if terrain_types:
        terrain_defaults.extend(terrain_types)

    feature_defaults = ["倒木 (Log)", "岩石 (Rock)", "泥泞区域 (Mud)"]
    if detected_semantics:
        for sem in detected_semantics:
            feature_defaults.append(f"{sem}")

    terrain_str = "、".join(terrain_defaults)
    feature_str = "、".join(feature_defaults)

    # -- Coordinate injection ------------------------------------------------------------
    coord_lines: List[str] = []
    for i, (x, y) in enumerate(trajectory_coords):
        coord_lines.append(f"路径点{i + 1}: ({x:.2f}, {y:.2f})")
    coord_block = "\n".join(coord_lines)

    # -- Assemble final prompt -----------------------------------------------------------
    prompt = f"""# 角色与目标
你是一个野外机器人导航专家。请结合提供的 BEV 图（几何结构）和 2D 语义图（类别特征），为机器人生成导航指令。{image_context}

# 地标参考词库 (Landmark Library)
地形 (Topological)：{terrain_str}。
群落 (Community)：密集林区、林缘交界、低矮灌丛带。
特征 (Feature)：{feature_str}。

# 指令编写约束
1. 几何优先：优先描述地形的变化（如：坡度起伏）和空间边界（如：路变宽/变窄）。
2. 避开主观颜色：禁止使用"红色的、深色的"等描述，改用语义类名或物理质感（如：硬质土路、松软草地）。
3. 动作简洁：仅使用"直行、左转、右转、微调航向"。
4. 排除内部术语：不要出现"锚点"这类复杂的词，确保指令在"色盲测试"下依然具备空间逻辑。

# 任务
先生成地标，再根据地标描述指令。
结合轨迹位移，{num_rgb_images}张图对应的{num_rgb_images}个路径点坐标如下，这些坐标并不是等时长选取的：
{coord_block}

请使用上述词库，严格按照 JSON 格式输出以下字段：
- extracted_landmarks: 列出你参考的地标列表（每个地标包含 name, category, role, relation 字段）。
- global_instruction: 【宏观指令】一句话描述起点到终点的地形转换和总体方向。
- meso_instruction: 【中观指令】识别路径中 1-2 个关键视觉锚点（如倒木或明显的树群），并说明其与路径的相对位置。
- micro_instruction: 【微观动作】描述未来 5-10 米内的具体动作序列，必须与坐标位移对齐。指令应该干练如："沿着当前道路直行，到岔路口左转或右转"。

只输出 JSON，不要包含其他文字。"""
    return prompt
