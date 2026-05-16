# 🤖 AI Captcha Recognizer

> **底层逻辑**：基于 MNIST 数据集与 CNN (卷积神经网络) 架构，实现对手写体数字验证码的高精度识别。通过深度学习模型赋能自动化交互场景。

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.0+-orange.svg)](https://tensorflow.org/)

---

## 🚀 核心抓手 (Key Features)
*   **CNN 模型驱动**：利用经典的 LeNet-5 变体架构，实现像素级特征提取。
*   **端到端识别**：从原始图片输入到结果输出的闭环流程。
*   **训练与预测分离**：提供 `ModelTraining.py` 进行离线训练，`Model_Identify.py` 实现快速推理。

## 🛠 技术实现 (Implementation)
1.  **数据清洗**：使用 OpenCV 对验证码进行去噪、二值化及字符切割。
2.  **模型构建**：两层卷积层 + 最大池化层 + 全连接层。
3.  **结果验证**：在测试集上实现 98%+ 的识别准确率。

## 📖 运行打法 (Usage)
1.  安装依赖：`pip install tensorflow opencv-python numpy`
2.  训练模型：`python ModelTraining.py`
3.  识别验证码：`python Model_Identify.py --image path/to/image.png`

---
*Created by Jacknie666*
