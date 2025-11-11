# MER2025
Multimodal Emotion Recognition 2025 Competition Learning

# reference
https://github.com/zeroQiaoba/MERTools/tree/master/MER2025

# 步骤
1. 在ubuntu物理机的anaconda内部署torch2.4.0+cu121
2. conda env create -f environment_vllm2.yml
3. conda activate vllm2



# 遇到的问题及解决方案
请尽量先保证系统级的cuda版本切换再进行环境部署
## ERROR: Ignored the following versions that require
- 从file environment_vllm2.yml构建过程中，pip安装的依赖几乎都被跳过了，因为我的conda-base环境并没有安装具体的runtime version<img src="https://github.com/fortitudelucifer/MER2025/blob/main/attachment/%E6%98%BE%E5%8D%A1-%E9%A9%B1%E5%8A%A8-GUI%E4%B9%8B%E9%97%B4%E7%9A%84%E5%85%B3%E7%B3%BB.png" height="50%" width="50%" >
  - 在channels里添加国内源
  - 在environment的文件中添加cuda-toolkit=12.1在vllm2环境中指定cuda版本安装，但pip因为没有cuda的原因导致environment文件后续的pip的一系列库/依赖没有安装，所以单独将它们提出来做成一个requirement.txt文件安装
  - 因为版本兼容问题pip中安装报错error故pip install单独安装了
    - bokeh
    - bottleneck
    - py-cpuinfo
    - pyairports
    - tool-helpers
    - opencv-python
    - scikit-image
    - scikit-learn
    - 最大的问题出现在flash-attn上，在我安装完所有的包后通过在 解决方案：https://blog.csdn.net/niuma178/article/details/135055359   Release v2.7.4.post1 · Dao-AILab/flash-attention 发布页直接下载 2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 文件然后在ubuntu里install成功 pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

## OSError（libtorchaudio.so undefined symbol）： torchaudio 与 torch 的二进制不匹配导致。solution：把torchaudio降级到2.40版本
- python -m pip install -U pip setuptools wheel
  - setuptool是一个Python库，旨在帮助开发者打包、分发和安装Python项目
- python -c "import sys; print('exe =', sys.executable)"
  - 确认 exe 指向 /home/forcifer-123/miniconda3/envs/vllm2/bin/python
- python -m pip uninstall -y torchaudio
  - 卸载冲突的 torchaudio
- python -m pip install --index-url https://download.pytorch.org/whl/cu121 torchaudio==2.4.0
  - 从 PyTorch 官方 cu121 源安装匹配的 torchaudio 2.4.0

## 担心关键换配置冲突，备份快照
- sudo apt update
- sudo apt install timeshift -y
- sudo timeshift --create --comments "安装CUDA前的备份" --tags D
  - First run mode (config file not found)
Selected default snapshot type: RSYNC
Mounted '/dev/nvme0n1p3' at '/run/timeshift/3925311/backup'
Selected default snapshot device: /dev/nvme0n1p3
  - Estimating system size...
Creating new snapshot...(RSYNC)
Saving to device: /dev/nvme0n1p3, mounted at path: /run/timeshift/3925311/backup
Syncing files with rsync...
Created control file: /run/timeshift/3925311/backup/timeshift/snapshots/2025-11-11_15-30-11/info.json
RSYNC Snapshot saved successfully (22s)

- sudo timeshift --list
  - 恢复备份
- sudo timeshift-gtk
  - 使用GUI
 
## 不同版本cuda的统一管理
其一参考：https://blog.csdn.net/beiyoulijun/article/details/146132289
### 系统级安装cuda13.0和cuda12.1
- mkdir -p ~/cuda_installers
- cd ~/cuda_installers/
- wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run
- sudo sh cuda_13.0.2_580.95.05_linux.run
- wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
- sudo sh cuda_12.1.0_530.30.02_linux.run
  - gcc不兼容，当前ubuntu gcc-version是13.3.0
- sudo apt update
- sudo apt install gcc-11 g++-11
- sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 
  - 将当前gcc-13设置稍低的优先级
- sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
- sudo update-alternatives --config gcc
 - 这个命令会列出所有已经“注册”的 gcc 选项，手动选择一个来作为当前的默认版本
- sudo sh cuda_12.1.0_530.30.02_linux.run
- A symlink already exists at /usr/local/cuda. Update to this installation?    │
│ Yes                                                                          │
│ No            选择No
- ls /usr/local | grep cuda
- sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-13.0 130
- sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.1 121
- sudo update-alternatives --config cuda **切换cuda**
- ls -l /usr/local/cuda
- /usr/local/cuda/bin/nvcc --version

\# 为了方便，可以将CUDA的 bin 和 lib64 目录添加到系统PATH。编辑 ~/.bashrc 或 /etc/profile：
- sudo tee /etc/profile.d/cuda.sh > /dev/null <<'EOF'
- export PATH=/usr/local/cuda/bin:$PATH
- export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
- EOF
- source /etc/profile.d/cuda.sh
## 为不同的anaconda环境配置特定cuda
- conda activate vllm2
- conda env config vars set CUDA_HOME=/usr/local/cuda-12.1
- conda env config vars set PATH=/usr/local/cuda-12.1/bin:$PATH
- conda env config vars set LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
- conda deactivate
- conda activate vllm2

### 环境设置崩掉了
- shutdown current terminal
- launch a new terminal
- conda env config vars unset PATH --name vllm2
- conda env config vars unset LD_LIBRARY_PATH --name vllm2
- conda env config vars unset CUDA_HOME --name vllm2
- conda env config vars list -n vllm2
  - 此时应该显示为空，没有任何输出。现在 vllm2 环境已经“干净”了，可以安全地激活了

### 重新设置环境变量
- mkdir -p ~/miniconda3/envs/vllm2/etc/conda/activate.d
- mkdir -p ~/miniconda3/envs/vllm2/etc/conda/deactivate.d
- nano ~/miniconda3/envs/vllm2/etc/conda/activate.d/env_vars.sh
  - 粘贴#!/bin/sh\# 记录原始的环境变量，以便退出时恢复
export OLD_CUDA_HOME=$CUDA_HOME
export OLD_PATH=$PATH
export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
\# 设置新的环境变量，明确指向 CUDA 12.1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
- nano ~/miniconda3/envs/vllm2/etc/conda/deactivate.d/env_vars.sh
  - 粘贴#!/bin/sh\# 恢复进入环境之前的环境变量
export CUDA_HOME=$OLD_CUDA_HOME
export PATH=$OLD_PATH
export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
\# 清理我们设置的临时备份变量
unset OLD_CUDA_HOME
unset OLD_PATH
unset OLD_LD_LIBRARY_PATH
- chmod +x ~/miniconda3/envs/vllm2/etc/conda/activate.d/env_vars.sh
- chmod +x ~/miniconda3/envs/vllm2/etc/conda/deactivate.d/env_vars.sh
### 重新激活vllm2环境和检查
- which nvcc
  - 检查 nvcc 命令的位置
- nvcc --version
  - 检查 nvcc 的版本
- echo $CUDA_HOME
  - 检查 CUDA_HOME 环境变量
### 环境再次崩溃
#### 首次激活 vllm2 成功：activate.d 脚本运行，将 CUDA 12.1 的路径添加到了 PATH 的最前面。此时一切正常。停用 vllm2 时灾难发生：deactivate.d 脚本运行。它执行了 export PATH=$OLD_PATH。但由于某些未知的 Shell 环境交互问题，此时的 $OLD_PATH 变量已经丢失或变为空了。结果：export PATH=$OLD_PATH 这个命令变成了 export PATH=，它直接将 PATH 环境变量设置成了一个空字符串。连锁反应：PATH 变空后，Shell 找不到任何系统命令，如 pip, which, grep, ls 等，因为存放它们的 /usr/bin, /bin 等目录已经不在“地址簿”里了。这就是“没有那个文件或目录”或“未找到命令”错误的根源。
- nano ~/miniconda3/envs/vllm2/etc/conda/activate.d/env_vars.sh 不用保存旧变量OLD，删掉保存即可
- nano ~/miniconda3/envs/vllm2/etc/conda/deactivate.d/env_vars.sh
  - #!/bin/sh\# 使用 sed 命令从 PATH 中安全地移除 CUDA 12.1 的 bin 目录。\# sed 的 's|...:||' 语法会查找并删除我们添加的路径和后面的冒号。
export PATH=$(echo "$PATH" | sed -e 's|/usr/local/cuda-12.1/bin:||')\# 同样地，从 LD_LIBRARY_PATH 中安全地移除 CUDA 12.1 的 lib64 目录。
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's|/usr/local/cuda-12.1/lib64:||')\# 取消设置 CUDA_HOME 变量
unset CUDA_HOME
- 终究还是坏掉了
- 备份以后重建环境 conda env export -n vllm2 > vllm2_environment.yml
### 重新安装vllm2环境
- conda env export -n vllm2 > environment.yml
- conda activate vllm2
- pip freeze > ~/requirements.txt
- conda deactivate
- conda env remove --name vllm2
- conda env create -f environment.yml
  - environment.yml 文件所在的目录下
- 不出意外flash-attn出问题了，先注释掉重新conda env create -f environment.yml
- 永久配置nvcc的环境变量
  - echo '' >> ~/.bashrc
  - echo '# Set default CUDA path' >> ~/.bashrc
  - echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  - echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  - source ~/.bashrc
- conda env create -f environment_vllm2.yml
  - 注释掉flash-attn/pytorch-lightning==2.5.0.post0niyuy
  - conda activate vllm2
  - pip install torch==2.4.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  - conda env create -f environment_vllm2.yml
  - 删除bokeh/bottleneck/py-cpuinfo/pyairports/debugpy/fonttools/jiter/propcache/protobuf后面的版本号
  - conda activate vllm2
  - 将pip的板块单独提出来成为 requirement.txt然后pip install -r requirements.txt遇到报错就删版本号再次运行
  - 安装flash-attn
  - bottleneck一直安装不上
  - 单独安装opencv-python/scikit-image/scikit-learn
  

