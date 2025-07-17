#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情识别模型训练脚本 - 修正版
基于FER2013数据集，使用CNN进行7种表情分类
最终输出TFLite格式模型
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 开始表情识别模型训练...")
print(f"TensorFlow版本: {tf.__version__}")
print(f"GPU设备: {tf.config.list_physical_devices('GPU')}")

class EmotionRecognitionTrainer:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def preprocess_data(self, csv_file_path):
        """预处理FER2013数据集"""
        print("正在加载和预处理数据...")
        start_time = time.time()
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        print(f"数据集大小: {len(df)} 个样本")
        
        # 解析像素数据
        pixels = df['pixels'].tolist()
        images = []
        
        for i, pixel_sequence in enumerate(pixels):
            if i % 5000 == 0:
                print(f"处理进度: {i}/{len(pixels)}")
            
            # 将字符串转换为numpy数组
            image = np.fromstring(pixel_sequence, dtype=int, sep=' ')
            # 重塑为48x48并归一化
            image = image.reshape(48, 48) / 255.0
            # 添加通道维度
            image = np.expand_dims(image, axis=-1)
            images.append(image)
        
        images = np.array(images)
        labels = df['emotion'].values
        
        # 标签编码为one-hot
        labels = to_categorical(labels, self.num_classes)
        
        print(f"图像形状: {images.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"数据预处理完成，耗时: {time.time() - start_time:.1f} 秒")
        
        return images, labels
    
    def create_model(self):
        """创建CNN模型架构"""
        print("创建模型架构...")
        
        model = tf.keras.Sequential([
            # 第一层卷积块
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # 第二层卷积块
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # 第三层卷积块
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # 全连接层
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            
            # 输出层
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        print("模型创建完成")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型编译完成")
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练模型"""
        print("开始训练模型...")
        
        # 数据增强
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 回调函数
        callbacks = [
            ModelCheckpoint(
                'best_emotion_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # 训练模型
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(X_val) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("训练完成")
        return history
    
    def evaluate_model(self, X_test, y_test):
        """评估模型性能"""
        print("评估模型性能...")
        
        # 预测
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # 计算准确率
        accuracy = np.mean(predicted_classes == true_classes)
        print(f"测试准确率: {accuracy:.4f}")
        
        # 分类报告
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n分类报告:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.emotion_labels))
        
        return accuracy
    
    def convert_to_tflite(self, model_path='best_emotion_model.h5', 
                         output_path='emotion_model.tflite', quantize=True):
        """转换模型为TFLite格式"""
        print("转换模型为TFLite格式...")
        
        # 加载最佳模型
        model = tf.keras.models.load_model(model_path)
        
        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            # 启用量化优化
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("启用模型量化...")
        
        # 转换
        tflite_model = converter.convert()
        
        # 保存
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # 检查模型大小
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"原始模型大小: {original_size:.2f} MB")
        print(f"TFLite模型大小: {tflite_size:.2f} MB")
        print(f"压缩比: {original_size/tflite_size:.2f}x")
        
        return output_path

def main():
    print("=" * 60)
    print("表情识别模型训练开始")
    print("=" * 60)
    
    # 初始化训练器
    trainer = EmotionRecognitionTrainer()
    
    # 预处理数据
    X, y = trainer.preprocess_data('archive/fer2013/fer2013.csv')
    
    # 分割数据集
    print("分割数据集...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建和编译模型
    model = trainer.create_model()
    trainer.compile_model()
    
    # 显示模型架构
    model.summary()
    
    # 训练模型
    history = trainer.train_model(
        X_train, y_train, X_val, y_val,
        epochs=50, batch_size=32
    )
    
    # 评估模型
    accuracy = trainer.evaluate_model(X_test, y_test)
    
    # 转换为TFLite
    tflite_path = trainer.convert_to_tflite()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"最佳模型已保存为: best_emotion_model.h5")
    print(f"TFLite模型已保存为: {tflite_path}")
    print(f"最终测试准确率: {accuracy:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main() 