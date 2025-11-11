# MER2025
Multimodal Emotion Recognition 2025 Competition Learning

# reference
https://github.com/zeroQiaoba/MERTools/tree/master/MER2025

# 步骤与思考
1.在ubuntu物理机的anaconda内部署torch2.4.0+cu121
2.conda env create -f environment_vllm2.yml
3.conda activate vllm2



# 遇到的问题及解决方案
## ERROR: Ignored the following versions that require
- 从file environment_vllm2.yml构建过程中，pip安装的依赖几乎都被跳过了，因为我的conda-base环境并没有安装具体的runtime version
<img src="https://github.com/fortitudelucifer/MER2025/blob/main/attachment/%E6%98%BE%E5%8D%A1-%E9%A9%B1%E5%8A%A8-GUI%E4%B9%8B%E9%97%B4%E7%9A%84%E5%85%B3%E7%B3%BB.png" height="50%" width="50%" >
  - 在channels里添加国内源
  - 在environment的文件中添加cuda-toolkit=12.1在vllm2环境中指定cuda版本安装，但pip因为没有cuda的原因导致environment文件后续的pip的一系列库/依赖没有安装，所以单独将它们提出来做成一个requirement.txt文件安装
  - 因为版本兼容问题pip中安装报错error故pip install单独安装了
    - bokeh
    - bottleneck
    - py-cpuinfo
    - pyairports
    - tool-helpers
    - opencv-python
    - 最大的问题出现在flash-attn上，在我安装完所有的包后通过在 Release v2.7.4.post1 · Dao-AILab/flash-attention 发布页直接下载 2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 文件然后在ubuntu里install成功

## OSError（libtorchaudio.so undefined symbol）： torchaudio 与 torch 的二进制不匹配导致。solution：把torchaudio降级到2.40版本
- python -m pip install -U pip setuptools wheel
  - setuptool是一个Python库，旨在帮助开发者打包、分发和安装Python项目
- python -c "import sys; print('exe =', sys.executable)"
  - 确认 exe 指向 /home/forcifer-123/miniconda3/envs/vllm2/bin/python
- python -m pip uninstall -y torchaudio
  - 卸载冲突的 torchaudio
- python -m pip install --index-url https://download.pytorch.org/whl/cu121 torchaudio==2.4.0
  - 从 PyTorch 官方 cu121 源安装匹配的 torchaudio 2.4.0


## ModuleNotFoundError: No module named 'sklearn'
- pip install scikit-learn

