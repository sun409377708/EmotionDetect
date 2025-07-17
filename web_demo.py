#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情识别Web演示应用
提供本地网页服务，支持图片上传和实时表情识别
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量
model = None
emotion_labels = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']
emotion_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8']

def load_model():
    """加载训练好的模型"""
    global model
    try:
        model = tf.keras.models.load_model("best_emotion_model.h5")
        print("✅ 模型加载成功")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def preprocess_image(image):
    """预处理图片"""
    try:
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用OpenCV的人脸检测
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # 使用检测到的第一个人脸
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face_coords = (x, y, w, h)
        else:
            # 如果没有检测到人脸，使用整张图片
            face = gray
            face_coords = None
        
        # 调整到48x48像素
        face_resized = cv2.resize(face, (48, 48))
        
        # 归一化
        face_normalized = face_resized / 255.0
        
        # 添加batch和channel维度
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        return face_input, face_coords
        
    except Exception as e:
        print(f"图片预处理错误: {e}")
        return None, None

def predict_emotion(image_data):
    """预测表情"""
    try:
        # 预处理图片
        processed_image, face_coords = preprocess_image(image_data)
        
        if processed_image is None:
            return None, None, None
        
        # 模型预测
        predictions = model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # 获取预测结果
        predicted_class = int(np.argmax(probabilities))  # 转换为Python int
        confidence = float(probabilities[predicted_class])
        
        # 构建结果
        result = {
            'predicted_emotion': emotion_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                emotion_labels[i]: float(probabilities[i]) 
                for i in range(len(emotion_labels))
            },
            'face_detected': face_coords is not None,
            'face_coords': tuple(map(int, face_coords)) if face_coords else None  # 转换为Python int
        }
        
        return result, probabilities, face_coords
        
    except Exception as e:
        print(f"预测错误: {e}")
        return None, None, None

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理图片上传和预测"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'})
        
        # 读取图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # 预测表情
        result, probabilities, face_coords = predict_emotion(image_array)
        
        if result is None:
            return jsonify({'error': '图片处理失败'})
        
        # 转换图片为base64用于显示
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if len(image_array.shape) == 3 else image_array
        
        # 如果检测到人脸，绘制人脸框
        if face_coords:
            x, y, w, h = face_coords
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_rgb, f"{result['predicted_emotion']}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', image_rgb)
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        result['image_base64'] = image_base64
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'处理错误: {str(e)}'})

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """静态文件服务"""
    return send_from_directory('static', filename)

def create_templates():
    """创建HTML模板"""
    # 创建模板目录
    os.makedirs('templates', exist_ok=True)
    
    # 创建主页模板
    html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>表情识别演示</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background-color: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background-color: #f0f4ff;
        }
        
        .upload-icon {
            font-size: 4em;
            color: #ddd;
            margin-bottom: 20px;
        }
        
        .upload-text {
            color: #666;
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
        }
        
        .result-area {
            display: none;
            margin-top: 30px;
        }
        
        .result-image {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .result-image img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .prediction-result {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .prediction-main {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .predicted-emotion {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence {
            font-size: 1.2em;
            color: #666;
        }
        
        .probabilities {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .prob-item {
            background: white;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .prob-label {
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .prob-bar {
            background: #f0f0f0;
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        
        .prob-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .prob-value {
            font-size: 0.9em;
            color: #666;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffe6e6;
            color: #d32f2f;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }
        
        .reset-btn {
            background: #6c757d;
            color: white;
            border: none;
            padding: 10px 25px;
            border-radius: 20px;
            cursor: pointer;
            margin-top: 20px;
        }
        
        .reset-btn:hover {
            background: #5a6268;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .probabilities {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 表情识别演示</h1>
            <p>上传一张人脸图片，让AI识别表情</p>
        </div>
        
        <div class="upload-section">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📷</div>
                <div class="upload-text">拖拽图片到这里或点击上传</div>
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    选择图片
                </button>
                <input type="file" id="fileInput" class="file-input" accept="image/*">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>正在分析图片...</div>
            </div>
            
            <div class="error" id="error"></div>
        </div>
        
        <div class="result-area" id="resultArea">
            <div class="result-image">
                <img id="resultImage" alt="上传的图片">
            </div>
            
            <div class="prediction-result">
                <div class="prediction-main">
                    <div class="predicted-emotion" id="predictedEmotion"></div>
                    <div class="confidence" id="confidence"></div>
                </div>
                
                <div class="probabilities" id="probabilities"></div>
            </div>
            
            <div style="text-align: center;">
                <button class="reset-btn" onclick="resetDemo()">重新上传</button>
            </div>
        </div>
    </div>

    <script>
        const emotions = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性'];
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8'];
        
        // 文件上传处理
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultArea = document.getElementById('resultArea');
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            // 检查文件类型
            if (!file.type.startsWith('image/')) {
                showError('请上传图片文件');
                return;
            }
            
            // 检查文件大小 (16MB)
            if (file.size > 16 * 1024 * 1024) {
                showError('文件太大，请上传小于16MB的图片');
                return;
            }
            
            uploadImage(file);
        }
        
        function uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // 显示加载状态
            hideError();
            showLoading();
            hideResult();
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
            })
            .catch(err => {
                hideLoading();
                showError('网络错误，请重试');
                console.error('Error:', err);
            });
        }
        
        function showResult(data) {
            // 显示图片
            document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.image_base64;
            
            // 显示预测结果
            const emotionIndex = emotions.indexOf(data.predicted_emotion);
            document.getElementById('predictedEmotion').textContent = data.predicted_emotion;
            document.getElementById('predictedEmotion').style.color = colors[emotionIndex];
            document.getElementById('confidence').textContent = `置信度: ${(data.confidence * 100).toFixed(1)}%`;
            
            // 显示概率分布
            const probContainer = document.getElementById('probabilities');
            probContainer.innerHTML = '';
            
            emotions.forEach((emotion, index) => {
                const prob = data.probabilities[emotion];
                const probItem = document.createElement('div');
                probItem.className = 'prob-item';
                
                probItem.innerHTML = `
                    <div class="prob-label" style="color: ${colors[index]}">${emotion}</div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${prob * 100}%; background-color: ${colors[index]}"></div>
                    </div>
                    <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                `;
                
                probContainer.appendChild(probItem);
            });
            
            resultArea.style.display = 'block';
        }
        
        function showLoading() {
            loading.style.display = 'block';
        }
        
        function hideLoading() {
            loading.style.display = 'none';
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
        
        function hideError() {
            error.style.display = 'none';
        }
        
        function hideResult() {
            resultArea.style.display = 'none';
        }
        
        function resetDemo() {
            hideResult();
            hideError();
            fileInput.value = '';
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML模板创建成功")

def main():
    """主函数"""
    print("🚀 启动表情识别Web演示服务")
    print("="*60)
    
    # 加载模型
    if not load_model():
        print("❌ 模型加载失败，无法启动服务")
        return
    
    # 创建模板
    create_templates()
    
    # 启动服务
    print("\n🌐 启动Web服务...")
    print("📍 本地地址: http://localhost:8080")
    print("📍 局域网地址: http://0.0.0.0:8080")
    print("\n💡 使用说明:")
    print("   1. 在浏览器中打开上述地址")
    print("   2. 上传包含人脸的图片")
    print("   3. 查看AI识别的表情结果")
    print("\n⛔ 按 Ctrl+C 停止服务")
    print("="*60)
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")

if __name__ == "__main__":
    main() 