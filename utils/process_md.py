import os
import re
from run import upload_image_to_picgo
import json


def process_markdown_file(md_file_path, prefix):
    # 读取Markdown文件
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 获取Markdown文件所在目录
    base_dir = os.path.dirname(md_file_path)

    # 使用正则表达式查找图片引用
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2)

        # 处理URL编码的路径
        image_path = image_path.replace("%20", " ")

        # 获取文件名并添加前缀
        image_name = os.path.basename(image_path)
        prefixed_image_name = f"{prefix}_{image_name}"

        # 构建完整的图片路径
        full_image_path = os.path.join(
            base_dir, os.path.dirname(image_path), prefixed_image_name
        )

        # 上传图片到PicGo
        response = upload_image_to_picgo(full_image_path)

        # 如果上传成功，替换图片链接
        if response and response.get("success"):
            new_url = response["result"][0]
            return f"![{alt_text}]({new_url})"

        # 如果上传失败，保持原样
        return match.group(0)

    # 替换所有图片引用
    new_content = re.sub(pattern, replace_image, content)

    # 写回文件
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


if __name__ == "__main__":
    md_file_path = "/Users/peyton/Downloads/run_notion2typora/48ac0b29-067b-44b0-a767-f02e6f59588e_Export-c4b0d895-d32e-4a11-a33f-b752028cd991/part2 1a1b5962cea4809da4b6c4c1023f5c1a.md"
    prefix = "part_2"  # 替换为实际使用的前缀
    process_markdown_file(md_file_path, prefix)
