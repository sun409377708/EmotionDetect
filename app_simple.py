#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版表情识别Web应用 - 用于测试部署
暂时不使用AI模型，确保部署成功
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 表情标签
emotion_labels = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求 - 模拟版本"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 读取图片
        image = Image.open(file.stream)
        
        # 模拟预测结果
        predicted_emotion = random.choice(emotion_labels)
        confidence = random.uniform(0.6, 0.95)
        
        # 创建模拟概率分布
        probabilities = {}
        remaining_prob = 1.0 - confidence
        for i, label in enumerate(emotion_labels):
            if label == predicted_emotion:
                probabilities[label] = confidence
            else:
                prob = random.uniform(0, remaining_prob / (len(emotion_labels) - 1))
                probabilities[label] = prob
                remaining_prob -= prob
        
        # 将图片转换为base64以便显示
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': probabilities,
            'image_base64': img_str,
            'note': '这是演示版本，使用随机结果'
        })
        
    except Exception as e:
        print(f"预测请求处理错误: {e}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': 'Demo',
        'note': '演示版本 - 使用随机结果'
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    print("🎭 启动演示版表情识别服务...")
    print(f"📍 访问地址: http://localhost:{port}")
    print("⚠️ 注意：这是演示版本，返回随机结果")
    app.run(host='0.0.0.0', port=port, debug=False) 