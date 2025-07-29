import sys
import click
import gc
import os
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
import psutil

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
    """高性能水印检测，优化CPU/RAM使用"""
    
    # 性能优化：限制图像尺寸
    max_size = 1024
    if max(image.width, image.height) > max_size:
        scale = max_size / max(image.width, image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 优化提示词列表
    text_inputs = ["watermark", "text overlay", "rotated text"]
    if sensitivity > 1.0:
        text_inputs.extend(["diagonal watermark", "logo"])
    
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    
    all_bboxes = []
    
    # 批量检测减少重复计算
    for text_input in text_inputs:
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            bboxes = parsed_answer[detection_key]["bboxes"]
            # 快速阈值过滤
            threshold = max(0.4, 0.6 / sensitivity)
            if "scores" in parsed_answer[detection_key]:
                bboxes = [bbox for bbox, score in zip(bboxes, parsed_answer[detection_key]["scores"]) 
                         if score > threshold]
            all_bboxes.extend(bboxes)
    
    # 智能旋转检测：仅对高敏感度启用，减少角度数量
    if sensitivity >= 1.2:
        angles = [-45, 45]  # 仅检测关键45度角
        for angle in angles:
            rotated_image = image.rotate(angle, expand=True)
            task_prompt = TaskType.OPEN_VOCAB_DETECTION
            parsed_answer = identify(task_prompt, rotated_image, "rotated watermark", model, processor, device)

            detection_key = "<OPEN_VOCABULARY_DETECTION>"
            if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
                all_bboxes.extend(parsed_answer[detection_key]["bboxes"])
    
    # 快速去重算法
    if all_bboxes:
        import numpy as np
        bboxes = np.array(all_bboxes)
        
        # 简化的NMS算法
        merged_bboxes = []
        if len(bboxes) > 0:
            # 按面积排序，优先大框
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            indices = np.argsort(areas)[::-1]
            
            used = set()
            for i in indices:
                if i in used:
                    continue
                
                x1_i, y1_i, x2_i, y2_i = bboxes[i]
                merged_bboxes.append([x1_i, y1_i, x2_i, y2_i])
                
                # 快速标记重叠框
                for j in indices:
                    if j in used:
                        continue
                    
                    x1_j, y1_j, x2_j, y2_j = bboxes[j]
                    overlap_x1 = max(x1_i, x1_j)
                    overlap_y1 = max(y1_i, y1_j)
                    overlap_x2 = min(x2_i, x2_j)
                    overlap_y2 = min(y2_i, y2_j)
                    
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        area_i = (x2_i - x1_i) * (y2_i - y1_i)
                        if overlap_area / area_i > 0.3:
                            used.add(j)
        
        image_area = image.width * image.height
        for bbox in merged_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                # 适度扩大检测框
                margin = max(2, int(min(x2-x1, y2-y1) * 0.1))
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.width, x2 + margin)
                y2 = min(image.height, y2 + margin)
                draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask



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

def apply_mosaic(image: Image.Image, mask: Image.Image, mosaic_size: int = 20):
    """
    对指定区域应用像素化马赛克效果
    
    Args:
        image: PIL Image对象
        mask: 需要像素化的区域掩码
        mosaic_size: 像素化块大小
    
    Returns:
        PIL Image: 处理后的图片
    """
    img_array = np.array(image)
    mask_array = np.array(mask.convert("L"))
    
    # 找到掩码中的非零区域
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        if w > 0 and h > 0:
            # 提取需要处理的区域
            roi = img_array[y:y+h, x:x+w]
            
            # 应用像素化：缩小再放大
            small = cv2.resize(roi, (max(1, w//mosaic_size), max(1, h//mosaic_size)), 
                              interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 将马赛克效果应用到原图
            img_array[y:y+h, x:x+w] = mosaic
    
    return Image.fromarray(img_array)

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--overwrite", is_flag=True, default=True, help="Overwrite existing files in bulk mode.")
@click.option("--transparent", is_flag=True, help="Make watermark regions transparent instead of removing.")
@click.option("--mosaic", is_flag=True, default=True, help="Apply mosaic effect to watermark regions instead of removing.")
@click.option("--mosaic-size", default=15, help="Mosaic pixel size (default: 15)")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["PNG", "WEBP", "JPG"], case_sensitive=False), default=None, help="Force output format. Defaults to input format.")
@click.option("--detection-sensitivity", default=1.7, help="Watermark detection sensitivity (0.5-2.0). Higher values detect more potential watermarks.")
@click.option("--include-rotated", is_flag=True, help="Include rotated text detection for better斜向水印检测.")
def main(input_path: str, output_path: str, overwrite: bool, transparent: bool, mosaic: bool, mosaic_size: int, max_bbox_percent: float, force_format: str, detection_sensitivity: float, include_rotated: bool):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 创建cache目录
    cache_dir = Path("C:/temp/wm_cache")
    cache_input_dir = cache_dir / "input"
    cache_output_dir = cache_dir / "output"
    
    cache_input_dir.mkdir(parents=True, exist_ok=True)
    cache_output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Script path: {os.path.abspath(__file__)}")
    
    # 检查是否在PyInstaller环境中运行
    if getattr(sys, 'frozen', False):
        logger.info("Running as compiled executable")
    else:
        logger.info("Running as script")
    
    # 检查模型文件是否存在
    model_name = "microsoft/Florence-2-large"
    logger.info(f"Loading model: {model_name}")
    
    florence_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    logger.info("Florence-2 Model loaded successfully")

    if not transparent and not mosaic:
        model_manager = ModelManager(name="lama", device=device)
        logger.info("LaMA model loaded")

    def handle_one(image_path: Path, output_path: Path):
        logger.info(f"Processing image: {image_path}")
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成cache中的文件名
        cache_input_file = cache_input_dir / image_path.name
        cache_output_file = cache_output_dir / image_path.name
        
        # 最终输出文件路径
        final_output_file = output_path / image_path.name
        
        if final_output_file.exists() and not overwrite:
            logger.info(f"Skipping existing file: {final_output_file}")
            return

        try:
            # 复制文件到cache目录
            import shutil
            shutil.copy2(image_path, cache_input_file)
            logger.info(f"Copied {image_path} to cache: {cache_input_file}")
            
            # 内存监控
            import psutil, gc
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            image = Image.open(cache_input_file).convert("RGB")
            original_size = image.size
            logger.info(f"Opened image: {image_path.name}, size: {image.size}")

            # 智能压缩大图像避免OOM
            max_input_size = 2048
            if max(image.width, image.height) > max_input_size:
                scale = max_input_size / max(image.width, image.height)
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
                logger.info(f"Resized image for processing: {image.size}")

            # 统一使用增强检测，减少分支
            logger.info("Starting watermark detection...")
            mask_image = get_enhanced_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent, detection_sensitivity)
            logger.info("Watermark detection completed")

            if mask_image.getbbox() is None:
                logger.warning("No watermark detected, saving original")
                if original_size != image.size:
                    image = image.resize(original_size, Image.Resampling.LANCZOS)
                image.save(output_path, quality=90)
                return

            logger.info("Applying watermark removal...")
            if transparent:
                result_image = make_region_transparent(image, mask_image)
                logger.info("Applied transparency effect")
            elif mosaic:
                result_image = apply_mosaic(image, mask_image, mosaic_size)
                logger.info(f"Applied mosaic effect with size {mosaic_size}")
            else:
                cv2_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(cv2_result, cv2.COLOR_BGR2RGB))
                logger.info("Applied LaMA inpainting")

            # 恢复原始尺寸
            if original_size != image.size:
                result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
                logger.info("Resized result image to original size")

            # 输出格式处理
            if force_format:
                output_format = force_format.upper()
            elif transparent:
                output_format = "PNG"
            else:
                output_format = image_path.suffix[1:].upper()
                if output_format not in ["PNG", "WEBP", "JPG"]:
                    output_format = "PNG"
            if output_format == "JPG":
                output_format = "JPEG"
            if transparent and output_format == "JPG":
                logger.warning("Transparency detected, using PNG")
                output_format = "PNG"
            logger.info(f"Output format determined: {output_format}")

            # 保存到cache输出目录
            cache_output_file = cache_output_file.with_suffix(f".{output_format.lower()}")
            result_image.save(cache_output_file, format=output_format, optimize=True)
            logger.info(f"Processed {cache_input_file} -> {cache_output_file}")
            
            # 复制处理后的文件到最终输出目录
            import shutil
            final_output_file = final_output_file.with_suffix(f".{output_format.lower()}")
            shutil.copy2(cache_output_file, final_output_file)
            logger.info(f"Copied result from cache: {final_output_file}")

            # 清理cache文件
            try:
                cache_input_file.unlink(missing_ok=True)
                cache_output_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean cache files: {e}")

            # 清理内存
            del image, mask_image, result_image
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB")

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            gc.collect()

    if input_path.is_dir():
        # 确保输出目录存在
        output_path.mkdir(parents=True, exist_ok=True)

        images = list(input_path.glob("*.[jp][pn]g")) + list(input_path.glob("*.webp"))
        total_images = len(images)

        for idx, image_path in enumerate(tqdm.tqdm(images, desc="Processing images")):
            handle_one(image_path, output_path)
            progress = int((idx + 1) / total_images * 100)
            print(f"input_path:{image_path}, output_path:{output_path}, overall_progress:{progress}")
    else:
        # 对于单个文件，output_path作为目录
        output_path.mkdir(parents=True, exist_ok=True)
        handle_one(input_path, output_path)
        print(f"input_path:{input_path}, output_path:{output_path}, overall_progress:100")

if __name__ == "__main__":
    main()
