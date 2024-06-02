#!/bin/bash

# 遍历所有的PNG文件
for f in *.png; do
    # 使用FFmpeg将PNG转换为GIF
    ffmpeg -i "$f" -vf "palettegen" -y "${f%.png}_palette.png"
    ffmpeg -i "$f" -i "${f%.png}_palette.png" -lavfi "paletteuse" -loop 0 "${f%.png}.gif"
done

