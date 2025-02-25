# 如何将 Markdown 文件转换为精美的 PDF 书籍

本文档介绍了如何使用 Pandoc 和 LaTeX 将 Markdown 文件转换为精美的 PDF 书籍。

## 前置要求

1. 安装 Pandoc：
```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc

# Windows
# 从 https://pandoc.org/installing.html 下载安装包
```

2. 安装 LaTeX：
```bash
# macOS
brew install --cask basictex    # 基础版本
# 或
brew install --cask mactex      # 完整版本

# Ubuntu/Debian
sudo apt-get install texlive-xetex texlive-lang-chinese

# Windows
# 安装 MiKTeX 或 TeX Live
```

## 文件结构

为了生成一本完整的书籍，你需要准备以下文件：

1. `metadata.yaml`：定义书籍的元数据和样式
2. `cover.md`：封面页面（可选）
3. 内容文件：你的 Markdown 文档

### metadata.yaml 示例

```yaml
---
title: '书籍标题'
subtitle: '副标题'
author: [作者名称]
date: \today
institute: '机构名称'
keywords: [关键词1, 关键词2]
subject: '主题'
lang: zh-CN
documentclass: ctexbook
classoption: 
  - UTF8
  - oneside
papersize: a4
linestretch: 1.25
fontsize: 11pt
mainfont: 'PingFang SC'
CJKmainfont: 'PingFang SC'
sansfont: 'PingFang SC'
monofont: 'Menlo'
geometry:
  - top=25mm
  - bottom=25mm
  - left=30mm
  - right=30mm
header-includes:
  - |
    ```{=latex}
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[L]{\leftmark}
    \fancyhead[R]{\rightmark}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0.4pt}
    ```
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
---
```

### cover.md 示例

```markdown
# 书籍标题

## 副标题

作者：作者名称

机构：机构名称

![封面图片](path/to/image.png)

\newpage
```

## 生成 PDF

使用以下命令生成 PDF：

```bash
pandoc metadata.yaml cover.md chapter1.md chapter2.md chapter3.md \
  -o book.pdf \
  --pdf-engine=xelatex \
  --top-level-division=chapter
```

## 常见问题解决

1. 中文字体问题：
   - 确保使用 `ctexbook` 或 `ctexart` 文档类
   - 指定中文字体，如 `PingFang SC`

2. 图片路径问题：
   - 使用相对路径或完整的 URL
   - 确保图片文件存在且有权限访问

3. 特殊字符问题：
   - 某些特殊字符（如表情符号）可能无法显示
   - 考虑使用替代字符或移除这些字符

## 优化建议

1. 排版优化：
   - 使用 `geometry` 包调整页边距
   - 设置适当的行间距 (`linestretch`)
   - 选择合适的字体大小

2. 目录优化：
   - 调整目录深度 (`toc-depth`)
   - 设置目录样式

3. 页眉页脚：
   - 使用 `fancyhdr` 包自定义页眉页脚
   - 添加页码和章节信息

4. 代码块：
   - 使用 `listings` 包美化代码显示
   - 设置适当的代码字体和颜色

## 高级功能

1. 交叉引用：
   - 使用 `\label{}` 和 `\ref{}` 进行引用
   - 添加图表编号

2. 参考文献：
   - 使用 BibTeX 管理参考文献
   - 添加引文和参考文献列表

3. 索引：
   - 使用 `makeindex` 生成索引
   - 添加关键词索引

## 参考资源

- [Pandoc 用户指南](https://pandoc.org/MANUAL.html)
- [LaTeX 文档](https://www.latex-project.org/help/documentation/)
- [CTeX 文档](https://mirrors.tuna.tsinghua.edu.cn/CTAN/language/chinese/ctex/ctex.pdf)
