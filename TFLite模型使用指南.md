# TFLite模型使用指南

## 📦 可用模型

当前目录包含以下TFLite模型：

| 模型文件 | 大小 | 推荐场景 | 特点 |
|---------|------|----------|------|
| `tflite_models/emotion_model_dynamic.tflite` | 0.68 MB | 🌐 **Web/服务器** | 最佳平衡 |
| `tflite_models/emotion_model_int8.tflite` | 0.69 MB | 📱 **移动端** | 最快速度 |
| `tflite_models/emotion_model_basic.tflite` | 2.61 MB | 🎯 **高精度** | 无损转换 |
| `tflite_models/emotion_model_float16.tflite` | 1.32 MB | 🔧 **GPU优化** | GPU友好 |
| `emotion_model.tflite` | 0.68 MB | 📜 **原有模型** | 向后兼容 |

## 🚀 快速选择指南

### 👉 推荐配置

**生产环境首选**: `emotion_model_dynamic.tflite`
- 最佳的性能/精度/大小平衡
- 推理时间: 0.90ms
- 精度损失: 微小

**移动应用首选**: `emotion_model_int8.tflite`  
- 最快推理速度: 0.84ms
- 最低内存占用
- 适合实时应用

## 💻 代码示例

### Python推理代码

```python
import tensorflow as tf
import numpy as np

def load_tflite_model(model_path):
    """加载TFLite模型"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_emotion(interpreter, image):
    """使用TFLite模型预测表情"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 预处理图像 (假设image是48x48的numpy数组)
    if input_details[0]['dtype'] == np.int8:
        # INT8模型需要特殊预处理
        input_data = ((image - 0.5) * 255).astype(np.int8)
    else:
        # Float32模型
        input_data = image.astype(np.float32)
    
    # 添加batch维度
    input_data = np.expand_dims(input_data, axis=0)
    
    # 推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # 处理输出
    if output_details[0]['dtype'] == np.int8:
        # INT8输出需要反量化
        scale, zero_point = output_details[0]['quantization']
        output_data = scale * (output_data.astype(np.float32) - zero_point)
    
    return output_data[0]  # 移除batch维度

# 使用示例
def main():
    # 选择模型 (根据需求更换路径)
    model_path = "tflite_models/emotion_model_dynamic.tflite"  # 推荐
    # model_path = "tflite_models/emotion_model_int8.tflite"   # 移动端
    
    # 加载模型
    interpreter = load_tflite_model(model_path)
    
    # 准备测试数据 (48x48灰度图像)
    test_image = np.random.random((48, 48, 1))
    
    # 预测
    predictions = predict_emotion(interpreter, test_image)
    
    # 解释结果
    emotions = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']
    predicted_emotion = emotions[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    print(f"预测表情: {predicted_emotion}")
    print(f"置信度: {confidence:.2%}")

if __name__ == "__main__":
    main()
```

### JavaScript/Web推理代码

```javascript
// 加载TensorFlow.js
// <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>

async function loadTFLiteModel(modelUrl) {
    // 使用动态量化模型 (推荐Web端)
    const model = await tf.loadLayersModel(modelUrl);
    return model;
}

function predictEmotion(model, imageData) {
    // imageData应该是48x48的图像数据
    const tensor = tf.browser.fromPixels(imageData, 1)  // 灰度
        .resizeNearestNeighbor([48, 48])
        .expandDims(0)
        .div(255.0);
    
    const predictions = model.predict(tensor);
    const probabilities = predictions.dataSync();
    
    const emotions = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性'];
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    
    return {
        emotion: emotions[maxIndex],
        confidence: probabilities[maxIndex],
        allProbabilities: probabilities
    };
}
```

## 📱 移动端集成

### Android (Kotlin)

```kotlin
// 使用INT8量化模型
class EmotionClassifier(context: Context) {
    private var interpreter: Interpreter
    
    init {
        val model = loadModelFile(context, "emotion_model_int8.tflite")
        interpreter = Interpreter(model)
    }
    
    fun predict(bitmap: Bitmap): String {
        val input = preprocessImage(bitmap)
        val output = Array(1) { ByteArray(7) }  // INT8输出
        
        interpreter.run(input, output)
        
        val emotions = arrayOf("愤怒", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性")
        val maxIndex = output[0].indexOf(output[0].maxOrNull()!!)
        return emotions[maxIndex]
    }
}
```

### iOS (Swift)

```swift
// 使用INT8量化模型
import TensorFlowLite

class EmotionClassifier {
    private var interpreter: Interpreter
    
    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "emotion_model_int8", ofType: "tflite") else {
            throw ClassifierError.invalidModel
        }
        interpreter = try Interpreter(modelPath: modelPath)
        try interpreter.allocateTensors()
    }
    
    func predict(image: UIImage) throws -> String {
        let inputData = preprocessImage(image)
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()
        
        let outputTensor = try interpreter.output(at: 0)
        let results = outputTensor.data
        
        let emotions = ["愤怒", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]
        // 处理INT8输出...
        return emotions[maxIndex]
    }
}
```

## ⚡ 性能对比

| 模型 | 推理速度 | 内存占用 | 精度 | 最适场景 |
|------|----------|----------|------|----------|
| Dynamic | 0.90ms | 低 | 高 | Web服务器 |
| INT8 | 0.84ms | 最低 | 高 | 移动应用 |
| Basic | 1.28ms | 中 | 最高 | 高精度需求 |
| Float16 | 1.02ms | 中低 | 高 | GPU推理 |

## 🔧 常见问题

### Q: 如何选择合适的模型？
**A**: 根据部署环境选择：
- Web/服务器: `emotion_model_dynamic.tflite`
- 手机App: `emotion_model_int8.tflite`
- 高精度场景: `emotion_model_basic.tflite`

### Q: INT8模型需要特殊处理吗？
**A**: 是的，INT8模型的输入输出都是int8格式，需要额外的量化/反量化步骤。

### Q: 哪个模型最小？
**A**: `emotion_model_dynamic.tflite` (0.68MB) 和 `emotion_model_int8.tflite` (0.69MB) 大小接近，都很适合移动端。

### Q: 精度会损失吗？
**A**: 所有量化模型的预测类别一致性都是100%，精度损失非常小(<4%)。

## 📚 更多资源

- [H5到TFLite兼容性分析报告](./H5到TFLite兼容性分析报告.md) - 详细的性能分析
- [模型准确性测试报告](./模型准确性测试报告.md) - 原始模型性能
- [Web演示应用](./web_demo.py) - 完整的Web应用示例

---

💡 **提示**: 建议优先使用 `emotion_model_dynamic.tflite`，它提供了最佳的性能、精度和大小平衡。 