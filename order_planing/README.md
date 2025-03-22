# Order Planing

## 功能

在每一个步骤规划拼装顺序。

## 准备工作

运行前需要设置环境变量：

- `OPENAI_API_KEY`
- `DINO_API_KEY`

搭建环境：

- python=3.9
- `pip3 install torch torchvision torchaudio openai opencv-python`
- Grounding-DINO API 改过一点，不能直接用 github 上的。使用 `pip install -v -e ./3rd_party/Grounding-DINO-1.5-API` 安装