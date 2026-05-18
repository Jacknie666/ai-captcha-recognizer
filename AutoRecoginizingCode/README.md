# ⚙️ AutoRecoginizingCode (验证码识别引擎核心组件)

> **定位与职责**：本目录为项目的核心算法与执行模块，封装了 CNN 模型结构定义、MNIST 高保真训练链路，以及基于 OpenCV 图像特征切割的手写体验证码端到端识别系统。

---

## 💻 运行环境与依赖配置 (System Prerequisites)

为了保障算法的高效运行与底层库的稳定性，请确保运行环境符合以下技术指标：

- **基础平台**：Python 3.8+ (推荐 3.9 或 3.10)
- **底层依赖库**：
  - **OpenCV (`opencv-python`)**：提供高保真的灰度转换、Otsu 自适应二值化、轮廓检测及边界计算支持。
  - **NumPy**：高效的矩阵运算与图像数组重整（Reshape/Normalize）抓手。
  - **TensorFlow / Keras**：提供经典卷积神经网络（CNN）的结构搭建与 GPU 硬件加速推理支持。
  - **scikit-learn**：提供 `train_test_split` 实现数据集的分层抽样与分发。

### 📦 极速部署（三步走打法）

1. **环境沙盒隔离（强烈推荐，防全局环境污染）**：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # Windows 环境请使用: .\venv\Scripts\activate
   ```

2. **安装核心库组件**：
   ```bash
   pip install --upgrade pip
   pip install opencv-python numpy tensorflow scikit-learn matplotlib
   ```

---

## 📖 核心组件打法与运行姿态 (Execution Architecture)

项目采取“**离线训练模型，在线实时推理**”的模块化架构。

### 1️⃣ 深度模型训练：`ModelTraining.py`
本脚本用于构建并训练卷积神经网络。
- **数据源**：自动拉取著名的 MNIST 手写体数据集，共 $60,000$ 张训练图，$10,000$ 张测试图。
- **数据分发**：以 $90\% : 10\%$ 的比例，利用 `stratify` 分层抽样策略切分出训练集与验证集，保持样本标签分布一致。
- **保存结果**：训练结束后最佳模型将保存为当前目录下的 `mnist_cnn_best_model.h5`。

```bash
# 启动离线模型高保真训练
python3 ModelTraining.py
```

### 2️⃣ 验证码端到端推理：`Model_Identify.py`
本脚本用于对待识别的验证码进行图像切割与单字元高精度预测。
- **默认测试图**：`custom_images/test1.png`
- **调用姿态**：
  1. 将测试图放入 `custom_images` 文件夹。
  2. 修改 `Model_Identify.py` 中的 `captcha_file_path` 为目标图路径。
  3. 执行识别脚本：
     ```bash
     python3 Model_Identify.py
     ```
  4. **交互确认**：程序在运行时会弹出 OpenCV 调试图像，**请在弹出的图片窗口激活的状态下，按键盘任意键**，程序才会继续向下运行并最终关闭所有窗口。

---

## 🔍 OpenCV 图像预处理与分割机制解析 (Image Processing Mechanics)

验证码分割的难点在于滤除杂色并完整保留单个字符的几何拓扑结构。本模块底层核心打法如下：

1. **灰度与二值化转换**：
   ```python
   gray_image = cv2.cvtColor(captcha_image_bgr, cv2.COLOR_BGR2GRAY)
   _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   ```
   * *注：`THRESH_BINARY_INV` 确保将原始偏暗的数字变为高亮白色，背景变为黑色，完全对齐 MNIST 的输入标准。*
2. **字符物理重排序（Left-to-Right Sorting）**：
   通过 `cv2.findContours` 提取出所有字符轮廓后，由于查找顺序是随机的，我们提取每个轮廓的 Bounding Box，并对其进行 X 轴坐标物理升序排序，打通从左到右拼接验证码的业务闭环：
   ```python
   bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
   ```
3. **等比例保留 Padding**：
   直接缩放非正方形 ROI 图像会导致数字字符严重变形。我们自动计算宽高差值，采用黑色背景像素进行对称扩充：
   ```python
   pad_y = int((max(w,h) - h)/2)
   pad_x = int((max(w,h) - w)/2)
   char_padded = cv2.copyMakeBorder(char_roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0])
   ```

---

## ⚠️ 避坑指南与技术边界 (Troubleshooting & Limitations)

- **窗口无响应/卡死**：
  - *原因*：OpenCV 的 `cv2.imshow` 窗口需要激活键盘监听。
  - *解决*：鼠标点击弹出的 `Thresholded CAPTCHA` 或 `Segmented Characters` 图片窗口，按键盘上的**空格键**或**任意字母键**即可顺利通过。
- **字母/复杂噪点识别错误**：
  - *局限*：由于底层分类器 Dense 层的输出神经元为 10 个（即对应数字 0-9），模型在逻辑上**无法识别任何英文字母、汉字或标点符号**。
  - *解决*：若需支持混合字母，需重新编写 `ModelTraining.py` 引入 EMNIST 或自定义字符集重新编译 Softmax 输出维度。
