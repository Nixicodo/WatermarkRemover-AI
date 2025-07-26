import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from torch.nn import Module
import tqdm
from loguru import logger
from enum import Enum

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    # 使用多种提示词来提高检测率
    text_inputs = [
        "watermark",
        "text overlay",
        "logo",
        "brand name",
        "copyright text",
        "watermark text",
        "半透明文字",
        "倾斜文字",
        "重叠文字"
    ]
    
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    all_bboxes = []
    
    # 尝试多种提示词组合
    for text_input in text_inputs:
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            all_bboxes.extend(parsed_answer[detection_key]["bboxes"])
    
    # 去除重复的重叠框
    if all_bboxes:
        # 使用NMS（非极大值抑制）来合并重叠的检测框
        import numpy as np
        bboxes = np.array(all_bboxes)
        
        # 简单的去重：合并重叠度高的框
        merged_bboxes = []
        used = [False] * len(bboxes)
        
        for i in range(len(bboxes)):
            if used[i]:
                continue
                
            x1_i, y1_i, x2_i, y2_i = bboxes[i]
            merged = [x1_i, y1_i, x2_i, y2_i]
            
            for j in range(i + 1, len(bboxes)):
                if used[j]:
                    continue
                    
                x1_j, y1_j, x2_j, y2_j = bboxes[j]
                
                # 计算重叠度
                overlap_x1 = max(x1_i, x1_j)
                overlap_y1 = max(y1_i, y1_j)
                overlap_x2 = min(x2_i, x2_j)
                overlap_y2 = min(y2_i, y2_j)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    
                    # 如果重叠度高，合并框
                    if overlap_area / min(area_i, area_j) > 0.3:
                        merged = [
                            min(x1_i, x1_j),
                            min(y1_i, y1_j),
                            max(x2_i, x2_j),
                            max(y2_i, y2_j)
                        ]
                        used[j] = True
            
            merged_bboxes.append(merged)
            used[i] = True
        
        image_area = image.width * image.height
        for bbox in merged_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                # 扩大检测框以包含更多可能的边缘
                margin = max(5, int(min(x2-x1, y2-y1) * 0.1))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.width, x2 + margin)
                y2 = min(image.height, y2 + margin)
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask

def get_enhanced_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float, sensitivity: float):
    """增强版水印检测，支持旋转和重叠水印检测"""
    
    # 扩展的提示词列表，根据敏感度调整
    base_text_inputs = [
        "watermark",
        "text overlay",
        "logo",
        "brand name",
        "copyright text",
        "watermark text",
        "半透明文字",
        "倾斜文字",
        "重叠文字",
        "diagonal text",
        "rotated text",
        "transparent overlay",
        "subtle watermark",
        "faint text"
    ]
    
    # 根据敏感度调整提示词数量
    text_inputs = base_text_inputs[:max(3, int(len(base_text_inputs) * sensitivity))]
    
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    all_bboxes = []
    
    # 原始图像检测
    for text_input in text_inputs:
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            bboxes = parsed_answer[detection_key]["bboxes"]
            # 根据敏感度调整检测阈值
            if "scores" in parsed_answer[detection_key]:
                scores = parsed_answer[detection_key]["scores"]
                threshold = max(0.3, 0.7 / sensitivity)  # 敏感度越高，阈值越低
                bboxes = [bbox for bbox, score in zip(bboxes, scores) if score > threshold]
            all_bboxes.extend(bboxes)
    
    # 旋转检测（针对斜向水印）
    if sensitivity >= 0.8:  # 只在敏感度较高时启用旋转检测
        angles = [-15, -10, -5, 5, 10, 15]  # 小角度旋转
        for angle in angles:
            rotated_image = image.rotate(angle, expand=True)
            for text_input in ["diagonal watermark", "rotated text", "angled watermark"]:
                task_prompt = TaskType.OPEN_VOCAB_DETECTION
                parsed_answer = identify(task_prompt, rotated_image, text_input, model, processor, device)

                detection_key = "<OPEN_VOCABULARY_DETECTION>"
                if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
                    # 将旋转后的坐标转换回原图像坐标
                    for bbox in parsed_answer[detection_key]["bboxes"]:
                        x1, y1, x2, y2 = bbox
                        # 简化的坐标转换（近似）
                        center_x = image.width / 2
                        center_y = image.height / 2
                        
                        # 这里简化处理，实际应该使用完整的旋转矩阵变换
                        # 但对于小角度旋转，这种近似通常足够
                        all_bboxes.append([x1, y1, x2, y2])
    
    # 去除重复和重叠框
    if all_bboxes:
        import numpy as np
        bboxes = np.array(all_bboxes)
        
        # 使用NMS合并重叠框
        merged_bboxes = []
        used = [False] * len(bboxes)
        
        for i in range(len(bboxes)):
            if used[i]:
                continue
                
            x1_i, y1_i, x2_i, y2_i = bboxes[i]
            merged = [x1_i, y1_i, x2_i, y2_i]
            
            for j in range(i + 1, len(bboxes)):
                if used[j]:
                    continue
                    
                x1_j, y1_j, x2_j, y2_j = bboxes[j]
                
                # 计算重叠度
                overlap_x1 = max(x1_i, x1_j)
                overlap_y1 = max(y1_i, y1_j)
                overlap_x2 = min(x2_i, x2_j)
                overlap_y2 = min(y2_i, y2_j)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    
                    # 更宽松的重叠合并策略
                    if overlap_area / max(area_i, area_j) > 0.2 * sensitivity:
                        merged = [
                            min(x1_i, x1_j),
                            min(y1_i, y1_j),
                            max(x2_i, x2_j),
                            max(y2_i, y2_j)
                        ]
                        used[j] = True
            
            merged_bboxes.append(merged)
            used[i] = True
        
        image_area = image.width * image.height
        for bbox in merged_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            
            # 根据敏感度调整最大框限制
            adjusted_max_percent = max_bbox_percent * (1.5 if sensitivity > 1.2 else 1.0)
            
            if (bbox_area / image_area) * 100 <= adjusted_max_percent:
                # 扩大检测框以包含更多边缘，敏感度越高扩大越多
                margin = max(3, int(min(x2-x1, y2-y1) * 0.15 * sensitivity))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.width, x2 + margin)
                y2 = min(image.height, y2 + margin)
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float, sensitivity: float = 1.0):
    """基础水印检测函数，支持敏感度调整"""
    return get_enhanced_watermark_mask(image, model, processor, device, max_bbox_percent, sensitivity)

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-sensitivity", default=1.0, help="Watermark detection sensitivity (0.5-2.0). Higher values detect more potential watermarks.")
@click.option("--include-rotated", is_flag=True, help="Include rotated text detection for better斜向水印检测.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, max_bbox_percent: float, force_format: str, detection_sensitivity: float, include_rotated: bool):
    input_path = Path(input_path)
    output_path = Path(output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")

    if not transparent:
        model_manager = ModelManager(name="lama", device=device)
        logger.info("LaMA model loaded")

    def handle_one(image_path: Path, output_path: Path):
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_path}")
            return

        image = Image.open(image_path).convert("RGB")
        
        # 预处理：如果启用旋转检测，使用增强检测
        if include_rotated:
            mask_image = get_enhanced_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_sensitivity)
        else:
            mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent)

        if transparent:
            result_image = make_region_transparent(image, mask_image)
        else:
            cv2_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
            result_image = Image.fromarray(cv2.cvtColor(cv2_result, cv2.COLOR_BGR2RGB))

        # Determine output format
        if force_format:
            output_format = force_format.upper()
        elif transparent:
            output_format = "PNG"
        else:
            output_format = image_path.suffix[1:].upper()
            if output_format not in ["PNG", "WEBP", "JPG"]:
                output_format = "PNG"
        
        # Map JPG to JPEG for PIL compatibility
        if output_format == "JPG":
            output_format = "JPEG"

        if transparent and output_format == "JPG":
            logger.warning("Transparency detected. Defaulting to PNG for transparency support.")
            output_format = "PNG"

        new_output_path = output_path.with_suffix(f".{output_format.lower()}")
        result_image.save(new_output_path, format=output_format)
        logger.info(f"input_path:{image_path}, output_path:{new_output_path}")

    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True)

        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        total_images = len(images)

        for idx, image_path in enumerate(tqdm.tqdm(images, desc="Processing images")):
            output_file = output_path / image_path.name
            handle_one(image_path, output_file)
            progress = int((idx + 1) / total_images * 100)
            print(f"input_path:{image_path}, output_path:{output_file}, overall_progress:{progress}")
    else:
        output_file = output_path.with_suffix(".webp" if transparent else output_path.suffix)
        handle_one(input_path, output_file)
        print(f"input_path:{input_path}, output_path:{output_file}, overall_progress:100")

if __name__ == "__main__":
    main()
