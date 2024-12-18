# **Watermark Removal Project**

## **模型水印移除项目**

---

### **Project Overview | 项目概述**
This project focuses on removing watermarks from images using advanced deep learning techniques. It employs a Generative Adversarial Network (GAN) architecture with residual blocks, attention mechanisms, and perceptual loss to produce clean and high-quality outputs.

此项目基于深度学习技术实现图像水印移除，采用生成对抗网络 (GAN) 架构，结合残差块、注意力机制以及感知损失，生成清晰且高质量的图像输出。

---

### **Features | 特点**

- **State-of-the-Art GAN Architecture**: Combines residual blocks, transformer blocks, and attention mechanisms.
  **先进GAN架构**：结合残差块、Transformer块和注意力机制。
- **Comprehensive Loss Functions**: Includes adversarial loss, perceptual loss, edge-preserving loss, and laplacian loss.
  **全面损失函数**：包括对抗损失、感知损失、边缘保护损失和拉普拉斯损失。
- **FastAPI Integration**: Provides an API for image processing via URLs or Base64 input.
  **集成FastAPI**：支持通过URL或Base64输入进行图像处理的API。
- **Batch Processing**: Supports batch image processing for enhanced efficiency.
  **批量处理**：支持批量图像处理以提高效率。

---

### **Project Structure | 项目结构**

```plaintext
.
|-- main.py          # Entry point for training and testing
|-- app.py           # FastAPI server for API integration
|-- dataset.py       # Custom dataset loader for watermarked images
|-- model.py         # Definitions of generator and discriminator models
|-- train.py         # GAN training pipeline
|-- test.py          # Testing and evaluation pipeline
|-- utils.py         # Utility functions for image processing
|-- models/          # Directory for saving trained models
|-- data/            # Directory for input datasets
|-- outputs/         # Directory for processed image outputs
```

---

### **File Explanations | 文件说明**

#### **main.py**
- **Description**: Entry point for training and testing the GAN model.
- **Key Parameters**:
  - `--mode`: Specifies the mode of operation (`train` or `test`).
    **模式**：指定操作模式（`train`训练，`test`测试）。
  - `--epochs`: Number of training iterations (default `100`).
    **迭代次数**：默认值为`100`。
  - `--model_path`: Path to the generator model file for testing.
    **模型路径**：测试时加载的生成器模型路径。
  - `--input_image` & `--output_image`: Paths for input and output images during testing.
    **输入和输出路径**：测试时指定的图像路径。

#### **app.py**
- **Description**: Implements a FastAPI server for real-time image processing.
- **Key Endpoints**:
  - `POST /process-image-url-preview/`: Accepts an image URL for watermark removal.
    **处理URL图像**：从URL加载图像并移除水印。
  - `POST /process-image-base64-preview/`: Accepts Base64-encoded image data.
    **处理Base64图像**：接收Base64编码的图像并移除水印。
- **Example**:
  ```json
  {
    "image_url": "http://example.com/image.jpg"
  }
  ```

#### **dataset.py**
- **Description**: Custom PyTorch dataset for training and validation.
- **Key Class**: `WatermarkDataset`.
  - **Methods**:
    - `__getitem__`: Returns a paired watermarked and clean image for training.
      **返回一对水印图像和干净图像，用于训练。**
    - `transform`: Resizes images to 256x256 and normalizes them to [-1, 1].
      **调整图像大小为256x256，并将其归一化到[-1, 1]范围内。**

#### **model.py**
- **Description**: Contains the definitions for the generator and discriminator.
- **Key Components**:
  - **Generator**: Utilizes residual blocks, attention mechanisms, and transformer layers to learn high-quality features.
    **生成器**：使用残差块、注意力机制和Transformer层以学习高质量特征。
  - **Discriminator**: Differentiates between real clean images and generated images.
    **判别器**：区分真实清晰图像与生成图像。
  - **Parameters**:
    - **Kernel Size**: Controls the receptive field in convolution layers.
      **卷积核大小**：决定卷积层的感受野。
    - **Channels**: Determines the depth of feature maps.
      **通道数**：控制特征图的深度。
    - **Transformer Block Size**: Default: 16. Ensures compatibility with input dimensions.
      **Transformer块大小**：默认值为16，确保与输入维度兼容。

#### **train.py**
- **Description**: Implements the GAN training pipeline.
- **Key Loss Functions**:
  - **Adversarial Loss**: Ensures the generator fools the discriminator.
    **对抗损失**：确保生成器可以欺骗判别器。
  - **Perceptual Loss**: Compares feature similarities between generated and clean images.
    **感知损失**：比较生成图像与清晰图像的特征相似性。
  - **Edge-Preserving Loss**: Focuses on retaining edges in the generated images.
    **边缘保护损失**：保留生成图像中的边缘。
  - **Laplacian Loss**: Enhances high-frequency components for sharpness.
    **拉普拉斯损失**：增强高频成分以提高图像清晰度。 

#### **test.py**
- **Description**: Provides testing and post-processing functionalities.
- **Key Steps**:
  1. Loads the trained generator model.
  2. Processes input images with padding for transformer compatibility.
  3. Applies post-processing (denoising, sharpening).

#### **utils.py**
- **Description**: Utility functions for image handling and preprocessing.
- **Key Functions**:
  - **load_image**: Loads an image and normalizes it to PyTorch tensor format.
    **加载图像并将其归一化为PyTorch张量格式。**
  - **pad_to_block_size**: Ensures input images are padded for transformer compatibility.
    **填充输入图像以确保与Transformer操作兼容。**

---

### **Key Parameters Influencing Output Quality | 影响输出质量的关键参数**

1. **Epochs | 训练轮次**:
   - Default: 100. More epochs improve quality but increase training time.
     **默认值：100。更多轮次会提升质量，但会增加训练时间。**

2. **Learning Rate | 学习率**:
   - Default: 0.0001. Higher rates may destabilize training; lower rates slow convergence.
     **默认值：0.0001。较高的学习率可能导致不稳定；较低的学习率会减慢收敛速度。**

3. **Batch Size | 批量大小**:
   - Default: 16. Larger sizes enhance stability but require more memory.
     **默认值：16。较大值提高稳定性，但需要更多内存。**

4. **Loss Weights | 损失权重**:
     g_loss = (
                    g_gan_loss +
                    9 * g_l1_loss +
                    1 * g_perceptual_loss +
                    3 * g_edge_loss +
                    1 * g_laplacian_loss +
                    1 * g_color_loss
                )
    - **GAN Loss (`g_gan_loss`) | GAN对抗损失**: Encourages realistic textures by ensuring generated images are indistinguishable from real ones.
    **驱动生成器学习真实纹理，使生成图像与真实图像难以区分。**

    - **L1 Loss (`g_l1_loss`) | L1损失**: Focuses on pixel-level accuracy to retain global structure. Weight: `10`.
    **在像素级保证准确性，帮助保留整体结构。权重：`9`。**

    - **Perceptual Loss (`g_perceptual_loss`) | 感知损失**: Preserves high-level semantic features using pre-trained networks like VGG19. Weight: `1`.
    **使用VGG19等预训练网络保留高层语义特征。权重：`1`。**

    - **Edge Loss (`g_edge_loss`) | 边缘损失**: Maintains sharp boundaries and prevents blurring. Weight: `2`.
    **保持边界清晰，防止模糊。权重：`3`。**

    - **Laplacian Loss (`g_laplacian_loss`) | 拉普拉斯损失**: Enhances details and sharpness by focusing on high-frequency components. Weight: `1`.
    **通过增强高频分量提高细节和清晰度。权重：`1`。**

    - **Color Loss (`g_color_loss`) | 色彩损失**: Ensures color consistency with the ground truth. Weight: `1`.
    **保证生成图像与真实图像的颜色一致性。权重：`1`。**


5. **Post-Processing | 后处理**:
   - Denoising: Reduces residual noise.
     **去噪**：减少残留噪声。
   - Sharpening: Enhances visual clarity.
     **锐化**：提升视觉清晰度。

---
### **Quick Start | 快速开始**

#### **Environment Setup | 环境配置**

1. **Install Required Packages | 安装必要的依赖**:
   Use the following command to install all required Python libraries:
   **使用以下命令安装所有必要的Python库：**
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include the following dependencies:
   **`requirements.txt` 文件应包含以下依赖：**
   ```plaintext
   torch
   torchvision
   fastapi
   uvicorn
   pillow
   opencv-python
   opencv-python-headless
   numpy
   ```

2. **Prepare Dataset | 准备数据集**:
   - Ensure the dataset is structured as follows:
     **确保数据集按照以下结构组织：**
     ```plaintext
     data/
     ├── train/
     │   ├── watermarked/   # Folder for input watermarked images
     │   ├── clean/         # Folder for corresponding clean images
     └── test/              # Folder for test images
     ```

#### **Training the Model | 训练模型**
```bash
python main.py --mode train --epochs 1500
```

1. Prepare the training dataset in `data/train/`.
   **准备训练数据集到 `data/train/` 文件夹中。**
2. Run the above command to train the model.
   **运行上述命令以训练模型。**
3. Check the `models/` directory for saved checkpoints.
   **在 `models/` 目录中检查保存的模型检查点。**

#### **Testing the Model | 测试模型**
```bash
python main.py --mode test --model_path models/generator_epoch_1500.pth --input_image input.jpg --output_image output.jpg
```
1. Ensure the trained model is in `models/`.
   **确保训练好的模型保存在 `models/` 中。**
2. Provide an input image and specify the output path.
   **提供输入图像并指定输出路径。**
3. Processed images will be saved at the output path.
   **处理后的图像将保存在输出路径中。**

#### **Running the FastAPI Server | 启动FastAPI服务**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
1. Start the server using the command above.
   **使用上述命令启动服务。**
2. Access API documentation at `http://localhost:8000/docs`.
   **在 `http://localhost:8000/docs` 访问API文档。**

---


### **Example Results | 示例结果**

#### Input Image (Watermarked):
![Input](data/test/sample_input.jpg)

#### Output Image (Cleaned):
![Output](outputs/sample_output.jpg)

---
