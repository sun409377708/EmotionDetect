#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情识别推理脚本
使用训练好的TFLite模型进行实时表情识别
支持摄像头实时检测和单张图片检测
"""

import cv2
import numpy as np
import tensorflow as tf
import argparse
import os
from pathlib import Path

class EmotionInference:
    def __init__(self, model_path='emotion_model.tflite'):
        """
        初始化表情识别推理器
        
        Args:
            model_path: TFLite模型文件路径
        """
        self.model_path = model_path
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_labels_zh = ['愤怒', '厌恶', '恐惧', '开心', '悲伤', '惊讶', '中性']
        
        # 加载TFLite模型
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出信息
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 加载人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print(f"模型已加载: {model_path}")
        print(f"输入尺寸: {self.input_details[0]['shape']}")
        print(f"输出尺寸: {self.output_details[0]['shape']}")
    
    def preprocess_face(self, face_img):
        """
        预处理人脸图像用于推理
        
        Args:
            face_img: 人脸图像 (OpenCV格式)
            
        Returns:
            处理后的图像数组
        """
        # 转换为灰度图
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # 调整大小到48x48
        resized = cv2.resize(gray, (48, 48))
        
        # 归一化
        normalized = resized / 255.0
        
        # 添加批次和通道维度
        input_data = np.expand_dims(normalized, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        
        return input_data.astype(np.float32)
    
    def predict_emotion(self, face_img):
        """
        预测单张人脸图像的表情
        
        Args:
            face_img: 人脸图像
            
        Returns:
            (predicted_emotion, confidence, all_probabilities)
        """
        # 预处理
        input_data = self.preprocess_face(face_img)
        
        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 推理
        self.interpreter.invoke()
        
        # 获取输出
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # 分析结果
        probabilities = output_data[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return predicted_class, confidence, probabilities
    
    def detect_and_predict(self, image):
        """
        检测图像中的人脸并预测表情
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像和检测结果
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = image[y:y+h, x:x+w]
            
            # 预测表情
            emotion_idx, confidence, probabilities = self.predict_emotion(face_roi)
            
            # 绘制框和标签
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 准备标签文本
            emotion_en = self.emotion_labels[emotion_idx]
            emotion_zh = self.emotion_labels_zh[emotion_idx]
            label = f"{emotion_zh}({emotion_en}): {confidence:.2f}"
            
            # 绘制标签
            cv2.putText(image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion_en,
                'emotion_zh': emotion_zh,
                'confidence': confidence,
                'probabilities': probabilities
            })
        
        return image, results
    
    def run_camera(self, camera_id=0):
        """
        运行摄像头实时表情识别
        
        Args:
            camera_id: 摄像头ID
        """
        print("开始摄像头实时表情识别...")
        print("按 'q' 退出, 按 's' 截图保存")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔几帧进行一次检测以提高性能
            if frame_count % 3 == 0:
                processed_frame, results = self.detect_and_predict(frame.copy())
                
                # 显示检测结果信息
                if results:
                    y_offset = 30
                    for i, result in enumerate(results):
                        info_text = f"人脸{i+1}: {result['emotion_zh']} ({result['confidence']:.3f})"
                        cv2.putText(processed_frame, info_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25
            else:
                processed_frame = frame
            
            # 显示FPS
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, processed_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Emotion Recognition', processed_frame)
            
            # 键盘事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emotion_capture_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"截图已保存: {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")
    
    def predict_image(self, image_path, save_result=True):
        """
        预测单张图片的表情
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果
        """
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        print(f"正在处理图片: {image_path}")
        
        # 检测和预测
        processed_image, results = self.detect_and_predict(image.copy())
        
        # 打印结果
        if results:
            print(f"检测到 {len(results)} 个人脸:")
            for i, result in enumerate(results):
                print(f"  人脸 {i+1}: {result['emotion_zh']}({result['emotion']}) - 置信度: {result['confidence']:.3f}")
                print(f"    所有概率: {dict(zip(self.emotion_labels_zh, result['probabilities']))}")
        else:
            print("未检测到人脸")
        
        # 显示图片
        cv2.imshow('Emotion Detection Result', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if save_result and results:
            output_path = f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(output_path, processed_image)
            print(f"结果已保存: {output_path}")
        
        return results
    
    def batch_predict(self, image_folder, output_folder=None):
        """
        批量预测文件夹中的图片
        
        Args:
            image_folder: 图片文件夹路径
            output_folder: 输出文件夹路径
        """
        if not os.path.exists(image_folder):
            print(f"文件夹不存在: {image_folder}")
            return
        
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        image_files = [f for f in os.listdir(image_folder) 
                      if Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"文件夹中没有找到图片文件: {image_folder}")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始批量处理...")
        
        results_summary = []
        
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"\n处理: {image_file}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"跳过无法读取的文件: {image_file}")
                continue
            
            processed_image, results = self.detect_and_predict(image.copy())
            
            # 记录结果
            file_result = {
                'filename': image_file,
                'face_count': len(results),
                'emotions': [r['emotion'] for r in results],
                'confidences': [r['confidence'] for r in results]
            }
            results_summary.append(file_result)
            
            # 保存处理后的图片
            if output_folder:
                output_path = os.path.join(output_folder, f"result_{image_file}")
                cv2.imwrite(output_path, processed_image)
            
            # 打印结果
            if results:
                emotions_str = ', '.join([f"{r['emotion_zh']}({r['confidence']:.2f})" for r in results])
                print(f"  检测结果: {emotions_str}")
            else:
                print("  未检测到人脸")
        
        # 打印总结
        print(f"\n批量处理完成!")
        print(f"总计处理 {len(results_summary)} 张图片")
        total_faces = sum(r['face_count'] for r in results_summary)
        print(f"总计检测到 {total_faces} 个人脸")
        
        if output_folder:
            print(f"结果已保存到: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='表情识别推理')
    parser.add_argument('--model', default='emotion_model.tflite', 
                       help='TFLite模型文件路径')
    parser.add_argument('--mode', choices=['camera', 'image', 'batch'], 
                       default='camera', help='运行模式')
    parser.add_argument('--input', help='输入图片路径或文件夹路径')
    parser.add_argument('--output', help='输出文件夹路径（批量模式）')
    parser.add_argument('--camera_id', type=int, default=0, help='摄像头ID')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 初始化推理器
    try:
        inference = EmotionInference(args.model)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 根据模式运行
    if args.mode == 'camera':
        inference.run_camera(args.camera_id)
    elif args.mode == 'image':
        if not args.input:
            print("图片模式需要指定 --input 参数")
            return
        inference.predict_image(args.input)
    elif args.mode == 'batch':
        if not args.input:
            print("批量模式需要指定 --input 参数")
            return
        inference.batch_predict(args.input, args.output)

if __name__ == "__main__":
    main() 