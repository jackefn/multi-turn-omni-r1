#!/bin/bash

# --- 配置区 ---
JSONL_FILE="examples/sft/qwen2.5_omni/example_data.jsonl" 

# 使用 export 将变量导出为环境变量，这样子 shell (xargs 启动的进程) 可以直接读取
# 确保路径以 / 结尾
export BASE_DIR="/mnt/hpfs/xiangc/mxy/sft-qwen2.5-omni-thinker/"

# --- 主执行区 ---

echo "Starting video preprocessing from ${JSONL_FILE}..."

# 1. 提取路径
# -r 参数非常重要，去掉 json 输出的双引号
VIDEO_PATHS=$(jq -r '.conversations[] | .content[] | select(.type == "video") | .video' "${JSONL_FILE}")

if [ -z "$VIDEO_PATHS" ]; then
    echo "Error: No video paths found."
    exit 1
fi

echo "Found $(echo "$VIDEO_PATHS" | wc -l) videos. Processing with 8 threads..."

# 2. 并行执行
# 我们不再通过参数传递 options，直接在 bash -c 内部写死或引用，避免转义错误
# xargs -I {} 将路径作为参数 $1 传递给 bash
echo "$VIDEO_PATHS" | xargs -P 8 -I {} bash -c '
    # 获取传入的相对路径 (即 xargs 的 {})
    REL_PATH="$1"
    
    # 忽略空行
    if [ -z "$REL_PATH" ]; then exit 0; fi

    # 构造绝对路径
    # 使用环境变量 BASE_DIR
    FULL_INPUT_PATH="${BASE_DIR%/}/${REL_PATH}"
    TEMP_OUTPUT_PATH="${FULL_INPUT_PATH}.temp.mp4"

    echo "[START] Processing: ${REL_PATH}"

    # 直接执行命令，不使用 eval，避免引号地狱
    # -y: 覆盖临时文件 (如果有)
    # -vf: 滤镜必须用引号包围
    # < /dev/null: 防止 ffmpeg 吞掉 stdin 导致 xargs 异常
    ffmpeg -y -hide_banner -loglevel error \
        -i "${FULL_INPUT_PATH}" \
        -vf "fps=3,scale=128:-1" \
        -c:v libopenh264 -b:v 1000k \
        -ar 16000 -ac 1 \
        "${TEMP_OUTPUT_PATH}" < /dev/null

    # 检查结果
    if [ $? -eq 0 ]; then
        # 成功：覆盖原文件
        mv -f "${TEMP_OUTPUT_PATH}" "${FULL_INPUT_PATH}"
        echo "[DONE] Finished: ${REL_PATH}"
    else
        echo "[ERROR] Failed: ${REL_PATH}"
        # 失败：清理临时文件
        rm -f "${TEMP_OUTPUT_PATH}"
    fi
' _ "{}" 

echo "--------------------------------------------------------"
echo "All tasks finished."