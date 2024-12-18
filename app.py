from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import base64
import requests
from io import BytesIO
from PIL import Image
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from scripts.model import EnhancedGenerator
import torchvision.transforms as transforms

# 初始化 FastAPI 应用
app = FastAPI(title="图片处理 API", description="支持通过图片链接和 Base64 编码处理图片，采用并行方式加速", version="1.0.0")

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = EnhancedGenerator().to(device)
generator.load_state_dict(torch.load("models/generator_epoch_1500.pth", map_location=device))
generator.eval()

# 图像转换函数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
unnormalize = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2))
])

# 定义输入数据结构
class ImageURLRequest(BaseModel):
    urls: List[str]  # 接收图片链接列表

class ImageBase64Request(BaseModel):
    images: List[str]  # 接收 Base64 图片列表

# 图片处理函数
def process_image(image: Image.Image) -> str:
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # 反归一化并转换为 numpy 格式
    output_tensor = unnormalize(output_tensor.squeeze(0)).clamp(0, 1)
    output_image_np = (output_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    output_image = Image.fromarray(output_image_np).resize(original_size, Image.LANCZOS)

    # 转换为 Base64
    buffer = BytesIO()
    output_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 并行处理图片链接
def process_url(url: str, headers: dict) -> str:
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"<p>无法访问图片链接: {url}，HTTP 状态码: {response.status_code}</p>"
        image = Image.open(BytesIO(response.content)).convert("RGB")
        processed_image_base64 = process_image(image)
        return f'<p>原始链接: {url}</p><img src="data:image/jpeg;base64,{processed_image_base64}" style="max-width: 500px; max-height: 500px;"/><hr>'
    except Exception as e:
        return f"<p>处理图片失败: {url}, 错误信息: {str(e)}</p><hr>"

# 并行处理 Base64 图片
def process_base64(image_base64: str) -> str:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        processed_image_base64 = process_image(image)
        return f'<img src="data:image/jpeg;base64,{processed_image_base64}" style="max-width: 500px; max-height: 500px;"/><hr>'
    except Exception as e:
        return f"<p>处理图片失败，错误信息: {str(e)}</p><hr>"

# 通过图片链接处理并提供预览（并行）
@app.post("/process-image-url-preview/", response_class=HTMLResponse)
async def process_image_url_preview(request: ImageURLRequest):
    start_time = time.time()  # 开始计时

    # 自定义 Headers
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "accept-language": "zh-CN,zh;q=0.9",
    }

    # 使用线程池并行处理图片链接
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda url: process_url(url, headers), request.urls))

    elapsed_time = time.time() - start_time  # 计算耗时
    return f"""
    <html>
    <body>
        <h1>图片处理结果</h1>
        {''.join(results)}
        <h3>处理完成，共耗时: {elapsed_time:.2f} 秒</h3>
    </body>
    </html>
    """

# 通过 Base64 编码处理并提供预览（并行）
@app.post("/process-image-base64-preview/", response_class=HTMLResponse)
async def process_image_base64_preview(request: ImageBase64Request):
    start_time = time.time()  # 开始计时

    # 使用线程池并行处理 Base64 图片
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_base64, request.images))

    elapsed_time = time.time() - start_time  # 计算耗时
    return f"""
    <html>
    <body>
        <h1>图片处理结果</h1>
        {''.join(results)}
        <h3>处理完成，共耗时: {elapsed_time:.2f} 秒</h3>
    </body>
    </html>
    """

# 根路径路由
@app.get("/")
async def root():
    return {"message": "欢迎使用图片处理 API，请访问 /docs 查看完整文档。"}
