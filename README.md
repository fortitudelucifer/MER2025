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
![image](https://github.com/fortitudelucifer/MER2025/blob/main/attachment/%E6%98%BE%E5%8D%A1-%E9%A9%B1%E5%8A%A8-GUI%E4%B9%8B%E9%97%B4%E7%9A%84%E5%85%B3%E7%B3%BB.png)

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

