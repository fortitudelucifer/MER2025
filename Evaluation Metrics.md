# WA（weighted accuracy）加权准确率

# UAR（unweighted average recall）未加权平均召回率

# F1-scoreF1分数是精确率（Precision）和召回率（Recall）的调和平均值

- grep -c "eval results" nohup.out  # 或 train.log # 方法1：统计日志中成功的epoch数
- for name in bloom-7b1-UTT chinese-macbert-large-UTT chinese-wav2vec2-large-UTT emonet_UTT videomae-base-UTT chinese-hubert-base-UTT chinese-roberta-wwm-ext-large-UTT clip-vit-base-patch32-UTT manet_UTT videomae-large-UTT chinese-hubert-large-UTT chinese-roberta-wwm-ext-UTT clip-vit-large-patch14-UTT resnet50face_UTT wavlm-base-UTT chinese-macbert-base-UTT chinese-wav2vec2-base-UTT dinov2-large-UTT senet50face_UTT whisper-large-v2-UTT; do
    count=$(grep -c "feature name: $name" nohup.out 2>/dev/null || echo 0)
    echo "$name: $count 次"
done # 方法2：统计每个特征完成的次数
- find ./saved-unimodal -name "*.pth" | wc -l  # 方法3：检查保存的模型数量
- grep "Loop index:" nohup.out | tail -10   # 方法3：检查保存的模型数量
