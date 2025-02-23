import re


def process_markdown_file(md_file_path):
    # 读取Markdown文件
    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 处理多行aside标签
    # 使用非贪婪匹配 (?s) 启用单行模式，使 . 可以匹配换行符
    pattern_multiline = r"<aside>\s*\n(.*?)\n\s*</aside>"

    def replace_multiline(match):
        # 获取aside标签内的内容
        content = match.group(1)
        # 分割成行
        lines = content.strip().split("\n")
        # 为每行添加 > 前缀，保持空行的格式
        quoted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                quoted_lines.append(f"> {line}")
            else:
                quoted_lines.append(">")
        # 用换行符连接所有行
        return "\n".join(quoted_lines)

    # 首先处理多行情况
    content = re.sub(pattern_multiline, replace_multiline, content, flags=re.DOTALL)

    # 处理单行aside标签
    pattern_singleline = r"<aside>(.*?)</aside>"

    def replace_singleline(match):
        # 获取aside标签内的内容并添加 > 前缀
        return f"> {match.group(1).strip()}"

    # 然后处理单行情况
    content = re.sub(pattern_singleline, replace_singleline, content)

    # 写回文件
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    # 示例用法
    md_file_path = "/Users/peyton/Downloads/run_notion2typora/48ac0b29-067b-44b0-a767-f02e6f59588e_Export-c4b0d895-d32e-4a11-a33f-b752028cd991/part2 1a1b5962cea4809da4b6c4c1023f5c1a.md"
    process_markdown_file(md_file_path)
