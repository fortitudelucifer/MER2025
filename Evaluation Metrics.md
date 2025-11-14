# WA（weighted accuracy）加权准确率

# UAR（unweighted average recall）未加权平均召回率

# F1-scoreF1分数是精确率（Precision）和召回率（Recall）的调和平均值

\# 检查所有特征文件夹是否存在
FEAT_DIR="/data/github_desktop/MERTools/MER2025/dataset/mer2025-dataset-process/features"

echo "=== 检查特征文件夹完整性 ==="

\# 定义所有特征名称
names=(
    'bloom-7b1-UTT'
    'chinese-macbert-large-UTT'
    'chinese-wav2vec2-large-UTT' 
    'emonet_UTT' 
    'videomae-base-UTT'
    'chinese-hubert-base-UTT' 
    'chinese-roberta-wwm-ext-large-UTT'
    'clip-vit-base-patch32-UTT' 
    'manet_UTT'
    'videomae-large-UTT' 
    'chinese-hubert-large-UTT' 
    'chinese-roberta-wwm-ext-UTT' 
    'clip-vit-large-patch14-UTT' 
    'resnet50face_UTT' 
    'wavlm-base-UTT' 
    'chinese-macbert-base-UTT' 
    'chinese-wav2vec2-base-UTT'
    'dinov2-large-UTT' 
    'senet50face_UTT' 
    'whisper-large-v2-UTT'
)

\# 统计变量
exists_count=0
missing_count=0

for name in "${names[@]}"; do
    if [ -d "$FEAT_DIR/$name" ]; then
        # 统计文件夹内的文件数量
        file_count=$(ls -1 "$FEAT_DIR/$name" 2>/dev/null | wc -l)
        folder_size=$(du -sh "$FEAT_DIR/$name" 2>/dev/null | cut -f1)
        echo "✅ $name (文件数: $file_count, 大小: $folder_size)"
        ((exists_count++))
    else
        echo "❌ $name (文件夹不存在)"
        ((missing_count++))
    fi
done

echo ""
echo "========================================="
echo "总计: ${#names[@]} 个特征"
echo "存在: $exists_count 个"
echo "缺失: $missing_count 个"
echo "========================================="
