import argparse
import os
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch

# Hi 2026

def parse_args():
    default_image = os.path.join(os.path.dirname(__file__), "12.png")
    parser = argparse.ArgumentParser(description="Hunyuan OCR")
    parser.add_argument(
        "image",
        nargs="?",
        default=default_image,
        help=f"Path to image file (default: {default_image})",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        parser.error(f"File not found: {args.image}")
    return args


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]

    return text


def main():
    img_path = parse_args().image

    model_name_or_path = "tencent/HunyuanOCR"
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
    image_inputs = Image.open(img_path)
    messages1 = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": (
                    "检测并识别图片中的文字，将文本坐标格式化输出。"
                )},
            ],
        }
    ]
    messages = [messages1]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto"
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        print("inputs: # fallback", inputs)
        input_ids = inputs.inputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = clean_repeated_substrings(processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ))
    print(output_texts)


if __name__ == "__main__":
    main()
