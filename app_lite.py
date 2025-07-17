#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级表情识别Web应用 - 专用于云部署
仅使用TFLite模型，减少依赖和内存占用
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量
model = None
emotion_labels = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']

def load_model():
    """加载TFLite模型"""
    global model
    try:
        model_path = "emotion_model.tflite"
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
            
        model = tflite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        print("✅ TFLite模型加载成功")
        return True
    except Exception as e:
        print(f"❌ TFLite模型加载失败: {e}")
        return False

def preprocess_image(image):
    """预处理图片"""
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 调整大小到48x48
        resized = cv2.resize(gray, (48, 48))
        
        # 归一化到0-1
        normalized = resized.astype('float32') / 255.0
        
        # 重塑为模型需要的形状 [1, 48, 48, 1]
        reshaped = normalized.reshape(1, 48, 48, 1)
        
        return reshaped, True
        
    except Exception as e:
        print(f"图像预处理错误: {e}")
        return None, False

def predict_emotion(image_array):
    """使用TFLite模型预测表情"""
    try:
        # 获取输入输出详情
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # 设置输入张量
        model.set_tensor(input_details[0]['index'], image_array.astype(np.float32))
        
        # 运行推理
        model.invoke()
        
        # 获取输出
        probabilities = model.get_tensor(output_details[0]['index'])[0]
        
        # 获取预测结果
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        emotion = emotion_labels[predicted_class]
        
        # 创建概率字典
        prob_dict = {}
        for i, label in enumerate(emotion_labels):
            prob_dict[label] = float(probabilities[i])
        
        return emotion, confidence, prob_dict
        
    except Exception as e:
        print(f"预测错误: {e}")
        return None, None, None

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 读取图片
        image = Image.open(file.stream)
        image_array = np.array(image)
        
        # 预处理
        processed_image, success = preprocess_image(image_array)
        if not success:
            return jsonify({'error': '图像预处理失败'}), 400
        
        # 预测
        emotion, confidence, probabilities = predict_emotion(processed_image)
        if emotion is None:
            return jsonify({'error': '表情识别失败'}), 500
        
        # 将图片转换为base64以便显示
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'predicted_emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'image_base64': img_str
        })
        
    except Exception as e:
        print(f"预测请求处理错误: {e}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'TFLite'
    })

# 初始化模型
print("正在初始化TFLite模型...")
if not load_model():
    print("❌ 模型初始化失败")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 