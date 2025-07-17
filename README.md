# 表情识别模型训练项目

基于FER2013数据集的深度学习表情识别系统，使用CNN模型进行7种基本表情分类，并支持转换为TFLite格式用于移动端部署。

## 📋 项目特性

- ✅ **7种表情识别**: 愤怒、厌恶、恐惧、开心、悲伤、惊讶、中性
- ✅ **高性能CNN架构**: 批量归一化 + Dropout + 数据增强
- ✅ **TFLite支持**: 支持模型量化和移动端部署
- ✅ **实时检测**: 摄像头实时表情识别
- ✅ **多种推理模式**: 单图片、批量处理、实时视频
- ✅ **详细评估**: 混淆矩阵、分类报告、训练曲线

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 验证TensorFlow安装
python -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__}')"
```

### 2. 数据集准备

确保您的FER2013数据集位于正确位置：
```
EmotionSecond/
├── archive/
│   └── fer2013/
│       └── fer2013.csv  # 35,888个样本
```

### 3. 训练模型

```bash
# 开始训练（需要较长时间，建议使用GPU）
python emotion_recognition_train.py
```

训练过程中会：
- 自动分割数据集（70%训练，15%验证，15%测试）
- 应用数据增强（旋转、缩放、翻转等）
- 保存最佳模型权重
- 生成训练曲线和混淆矩阵
- 自动转换为TFLite格式

### 4. 模型推理

训练完成后，您会得到以下文件：
- `best_emotion_model.h5` - 最佳Keras模型
- `emotion_model.tflite` - TFLite模型（用于部署）
- `training_history.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵

#### 摄像头实时检测
```bash
python emotion_inference.py --mode camera
```

#### 单张图片检测
```bash
python emotion_inference.py --mode image --input your_image.jpg
```

#### 批量图片处理
```bash
python emotion_inference.py --mode batch --input images_folder --output results_folder
```

## 📊 预期性能

根据我们的研究和类似项目，您可以期待：

- **训练准确率**: 80-85%
- **验证准确率**: 65-75%
- **测试准确率**: 63-70%
- **TFLite模型大小**: 约2-5MB（量化后）
- **推理速度**: 移动端10-50ms每张图片

### 性能对比
| 数据集 | 模型类型 | 准确率 | 模型大小 |
|--------|----------|--------|----------|
| FER2013 | 基础CNN | ~66% | ~10MB |
| FER2013 | 优化CNN | ~70% | ~5MB |
| FER2013 | TFLite量化 | ~68% | ~2MB |

## 🏗️ 模型架构

```
输入: 48x48x1 灰度图像
├── Conv2D(32) + BN + ReLU
├── Conv2D(32) + BN + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(64) + BN + ReLU  
├── Conv2D(64) + BN + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(128) + BN + ReLU
├── Conv2D(128) + BN + ReLU + MaxPool + Dropout(0.25)
├── Flatten
├── Dense(512) + BN + ReLU + Dropout(0.5)
├── Dense(256) + BN + ReLU + Dropout(0.5)
└── Dense(7) + Softmax
输出: 7类表情概率
```

## 🔧 高级配置

### 自定义训练参数

编辑 `emotion_recognition_train.py` 中的参数：

```python
# 训练配置
epochs = 100          # 训练轮数
batch_size = 32      # 批大小
learning_rate = 0.001 # 学习率

# 数据增强配置
rotation_range = 20      # 旋转角度
width_shift_range = 0.2  # 水平移动
height_shift_range = 0.2 # 垂直移动
zoom_range = 0.2        # 缩放范围
```

### TFLite量化选项

```python
# 在convert_to_tflite方法中
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 启用量化
# 或者使用更激进的量化
converter.target_spec.supported_types = [tf.float16]
```

## 📁 项目结构

```
EmotionSecond/
├── emotion_recognition_train.py  # 主训练脚本
├── emotion_inference.py          # 推理脚本
├── requirements.txt              # 依赖列表
├── README.md                     # 项目文档
├── archive/
│   └── fer2013/
│       └── fer2013.csv          # 数据集
├── best_emotion_model.h5        # 训练好的模型
├── emotion_model.tflite         # TFLite模型
├── training_history.png         # 训练曲线
└── confusion_matrix.png         # 混淆矩阵
```

## 🎯 使用建议

### 提高性能的技巧

1. **数据平衡**: FER2013数据集不平衡，考虑使用类权重
2. **数据增强**: 增加更多数据增强技术
3. **模型集成**: 训练多个模型进行投票
4. **预训练**: 使用预训练的CNN backbone
5. **超参数调优**: 使用Optuna等工具自动调参

### 部署优化

1. **量化**: 使用int8量化减小模型大小
2. **模型剪枝**: 移除不重要的连接
3. **知识蒸馏**: 用大模型训练小模型
4. **边缘优化**: 针对特定硬件优化

## ⚠️ 注意事项

1. **计算资源**: 完整训练需要2-4小时（GPU）或8-12小时（CPU）
2. **内存需求**: 至少8GB RAM，建议16GB
3. **数据集**: 确保FER2013数据集完整（35,888个样本）
4. **版本兼容**: 建议使用TensorFlow 2.12+

## 🐛 常见问题

### Q: 训练过程中出现内存错误
A: 减小batch_size或使用梯度累积

### Q: TFLite转换失败
A: 检查TensorFlow版本，确保使用兼容的操作

### Q: 摄像头无法打开
A: 检查摄像头权限和设备ID

### Q: 准确率较低
A: 增加训练时间、调整学习率或增强数据增强

## 📚 参考资料

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Deep Learning for Emotion Recognition](https://arxiv.org/abs/1312.6199)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

🚀 **开始您的表情识别之旅！**运行训练脚本，几小时后您就能拥有自己的表情识别模型！ 