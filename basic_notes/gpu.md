# GPU

> 还未测试，只是简单的记录下

## 1. 软件

必须要按照官网上的软件要求来安装

**2019/01/18** 

- NVIDIA GPU 驱动程序：最新版本即可 `(>=384.x)` 
- CUDA：CUDA **9.0** 
- cuDNN：最新版本即可 `(>=7.2)` 

> 如果需要支持其他版本的CUDA，需要自行从源码编译

## 2. 设置

将 CUDA、CUPTI 和 cuDNN 安装目录添加到 `%PATH%` 环境变量中。例如，如果 CUDA 工具包安装到了 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0` 并且 cuDNN 安装到了 `C:\tools\cuda`，请更新 `%PATH%` 以匹配路径：

```
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%
```

## 3. tensorflow-gpu

```bash
pip install tensorflow-gpu
```

