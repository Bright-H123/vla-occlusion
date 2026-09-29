# SmolVLA 在 Windows + RTX 4060 上的部署指南

面向从未接触 VLA 的读者，针对 MuJoCo 仿真中以末端增量为动作、以 1 到 2 路 RGB 相机加本体感受向量为观测的操作任务。动作维度按 6 到 14 维讨论，本仓库的推方块任务是 3 维，同样适用。

本文所有命令和数字都在这台机器上实际跑过。凡是没有实测的地方，都写明"未实测"。

验证环境：Windows 11 Home China 26100，RTX 4060 Laptop 8188 MiB，driver 560.94，无 WSL，无 nvcc，无 VS Build Tools。conda 在 `D:\cuda`。验证用环境 `D:\cuda\envs\vlax`，Python 3.12.14，torch 2.11.0+cu126，lerobot 0.6.1，transformers 5.5.4，mujoco 3.14.0，numpy 2.2.6。

---

## 0. 先读这一节：VLA 是什么，SmolVLA 到底在算什么

### 0.1 和你已有的知识对照

VLA 指 vision language action 模型。它的接口是：输入若干路 RGB 图像、一句自然语言指令、一个本体感受向量，输出一段未来动作序列。SmolVLA 一次输出 50 步动作，这叫 action chunk。

它不是控制器，不做模型预测，不求解优化问题。它是行为克隆，也就是监督学习：你先录一批示教轨迹，模型学习"在这个画面、这句指令、这个关节状态下，专家接下来 50 步会怎么动"。训练时没有奖励，没有与环境交互，没有探索。这和你熟悉的 TD3 完全不同，更接近你用 IK 生成一批轨迹然后拟合一个回归器。

因此第一条判断：SmolVLA 不会凭空学会你的任务。你必须先有能完成任务的示教数据。`D:\vla-occlusion` 里那个 scripted expert 正是这类数据的来源，手机 IMU 遥操作是另一类来源。

第二条判断：如果你的任务只有一个固定目标，scripted expert 已经能解，那 SmolVLA 在这个任务上不会比它更好。VLA 的价值在语言条件化（同一套权重根据指令切换"推门"和"推箱子"）和对新初始条件的泛化。这两点本次没有实测，只验证了训练和推理的工程链路通畅。

### 0.2 SmolVLA 的内部结构，读源码与实测得到

模型由两部分组成。

前半是一个视觉语言模型 SmolVLM2-500M-Video-Instruct，负责把图像和指令编码成一串 token。lerobot 只保留它的前 16 层。这部分在微调时整体冻结并置于 eval 状态。

后半是 action expert，一个宽度较小的 transformer，通过 cross attention 读取前半的 KV cache，用 flow matching 从高斯噪声出发回归出动作。这部分是训练时唯一更新的主体。

实测参数量：总计 450,046,176，可训练 99,880,992。可训练部分是 lm_expert 的 98,245,840 加上六组小投影层：`state_proj`、`action_in_proj`、`action_out_proj`、`action_time_mlp_in`、`action_time_mlp_out`，共十个张量。

一次推理的数据流，实测确认：

```
observation.images.<cam>   (B,3,H,W) float32 in [0,1]
        -> resize_with_pad 到 512x512 (padding 加在左边和上边)
        -> img*2-1 映射到 [-1,1]
        -> 每路相机编码成 64 个 token，宽度 960
task "push the door open"
        -> 自动追加 "\n"
        -> SmolVLM2 tokenizer，max_length=48，right padding
observation.state (B,d) -> 右侧零填充到 32 -> Linear(32,960) -> 1 个 token

prefix 长度 = 64 * 相机数 + 语言 token 数 + 1
        (实测：1 路相机、5 个语言 token 时 prefix 为 (1,70,960))

然后：noise ~ N(0,1) 形状 (B,50,32)
        -> forward Euler 积分 10 步，t = 1.0, 0.9, ..., 0.1
        -> 输出 (B,50,32)，按真实动作维度切片回 (B,50,d)
```

几个由此推出的工程结论。

图像必须是 float32 且落在 [0,1]。模型内部不做 dtype 转换，uint8 输入的后果见第 7 节。

`n_obs_steps` 对 SmolVLA 是失效参数。`observation_delta_indices` 硬编码为 `[0]`，内部队列只缓存 ACTION，不缓存观测历史。设成 2 不报错也不起作用。

一次推理产出 50 步动作，`n_action_steps` 决定队列里存几步。存 50 步就意味着连续 50 个控制周期不看相机，在 30 fps 下是 1.67 s 开环。推门这种有接触的任务需要把它降下来，代价见第 3 节的实测控制率。

### 0.3 SmolVLA 与 ACT 的取舍

同一个数据集上两者都跑过，实测对比在第 5 节。概括起来，SmolVLA 有语言条件化，参数量是 ACT 的 8.7 倍，单步训练慢 3.5 倍，单次推理慢 30 倍，显存约 1.9 倍。ACT 没有语言输入，一个 checkpoint 只对应一个任务。

建议的顺序是先用 ACT 把数据管线和 MuJoCo 环境跑通，因为它训练 13 s 就出结果，排错快。管线确认无误后再换 SmolVLA，这时你面对的只剩权重和特征命名的问题。

---

## 1. 环境准备

### 1.1 为什么要新建环境

现有的 `D:\cuda\envs\vla` 里 numpy 是 2.5.3，而 lerobot 0.6.1 声明 `numpy<2.3.0,>=2.0.0`，装不进去。现有的 `D:\cuda\envs\bedbot` numpy 是 2.2.6，本来兼容，但它缺 transformers 和 num2words，且是别的项目在用。所以另建一个。

关于 numpy 版本的判断依据是直接读 wheel metadata 得到的，不是猜的。mujoco 3.14.0 的 METADATA 里写的是 `Requires-Dist: numpy`，没有上界。lerobot 0.6.1 的 PKG-INFO 写的是 `numpy<2.3.0,>=2.0.0`。2.2.6 同时满足两者，所谓"mujoco 和 lerobot 的 numpy 冲突"并不存在。

lerobot 0.6.1 还要求 `packaging<26.0,>=24.2` 和 `setuptools<82.0.0,>=71.0.0`，以及 Python >= 3.12。3.12 是下限不是建议。

### 1.2 为什么是 cu126 而不是 cu128

lerobot 自己的 pyproject.toml 里写着 cu128 的 driver 下限是 570.86，而这台机器的 driver 是 560.94，低于下限。同一个文件里 `[tool.uv.sources]` 只在 `sys_platform == 'linux'` 时把 torch 指向 cu128 索引，所以在 Windows 上直接 pip install torch 会静默拿到 CPU 版。

实测证据：PyPI 上 torch 2.11.0 的 cp312 win_amd64 wheel 是 114,556,047 字节，而 cu126 的同名 wheel 是 2596.5 MB。前者就是 CPU 版。

### 1.3 完整命令

在 PowerShell 里逐条执行。

```powershell
# 1) 建环境，实测 27.5 s
D:\cuda\Scripts\conda.exe create -n vlax python=3.12 -y

# 2) CUDA 版 torch，实测约 8 分钟，下载 2621.4 MB
D:\cuda\envs\vlax\python.exe -m pip install "torch==2.11.0+cu126" "torchvision==0.26.0+cu126" --extra-index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) lerobot 本体加 smolvla 与 training 两组 extras
D:\cuda\envs\vlax\python.exe -m pip install "lerobot[smolvla,training]==0.6.1" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) MuJoCo
D:\cuda\envs\vlax\python.exe -m pip install "mujoco==3.14.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

第 2 条里的 `--extra-index-url` 不能换成 `-f` 或 `--find-links`。实测用 `-f https://download.pytorch.org/whl/cu126` 会得到：

```
ERROR: Could not find a version that satisfies the requirement torch==2.11.0+cu126
(from versions: 2.2.0, ... 2.11.0, 2.12.0, 2.13.0, 2.14.0)
ERROR: No matching distribution found for torch==2.11.0+cu126
```

同一条命令换成 `--extra-index-url` 立刻成功。

第 3 条在实测时是分两次执行的：`lerobot[smolvla]` 用 83.6 s 下载 100.1 MB，随后补装 training extras 用 55.5 s 下载 102.8 MB。合成一条命令的写法没有单独计时，但依赖解析结果相同。这里必须装 training，因为 `lerobot[smolvla]` 只解析出 transformers、num2words、accelerate 三项，`datasets`、`pandas`、`pyarrow`、`av`、`jsonlines`、`wandb` 全部缺失，`lerobot-train` 会直接报错。

第 4 条实测 51 s 下载 4.3 MB，但那次 18.5 MB 的 wheel 已经在 pip HTTP cache 里，干净机器上按 23 MB 估算。

可选的第 5 条，装 conda-forge 的 ffmpeg。它并不能修好 torchcodec（见 1.5），装了只多占 64.6 MB，可以跳过。

```powershell
$env:PYTHONUTF8="1"
D:\cuda\Scripts\conda.exe install -n vlax -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ ffmpeg=7.1.1 -y
```

`$env:PYTHONUTF8="1"` 是必需的。不设它这条命令在中文 locale 下会失败并回滚整个事务，错误是 `UnicodeDecodeError('gbk', ...)`，来自 gdk-pixbuf 的 post-link 脚本。

### 1.4 一条建议的改动

实测的第 2 条命令没有 pin numpy，结果是 torch 步先装 numpy 2.5.3，第 3 条再卸掉重下 2.2.6，packaging 也从 26.3 降到 25.0，白跑一遍下载。把第 2 条写成下面这样可以省掉这次降级。

```powershell
D:\cuda\envs\vlax\python.exe -m pip install "torch==2.11.0+cu126" "torchvision==0.26.0+cu126" "numpy==2.2.6" --extra-index-url https://download.pytorch.org/whl/cu126 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

加 pin 这一版没有单独实测，但最终解析出的 numpy 版本就是 2.2.6，与实测结果一致。

### 1.5 镜像与耗时

镜像选择对那个 2.6 GB 的 torch wheel 影响很大，实测数据如下。

| 源 | 实测速度 |
| --- | --- |
| pypi.tuna.tsinghua.edu.cn | 36.2 MB/s |
| download.pytorch.org 走本机代理 127.0.0.1:7890 | 6.2 MB/s |
| mirrors.aliyun.com/pytorch-wheels | 0.75 MB/s |

不要用 aliyun 的 pytorch-wheels 镜像，实测 22 分钟只下了 893 MB 就放弃了。清华、BFSU、NJU、CERNET 都没有 pytorch-wheels 镜像，全部返回 404，只有 aliyun 有，而它慢。所以 torch 只能走 download.pytorch.org，其余包走清华。

总耗时与体积，实测：conda create 27.5 s，torch 约 8 min，lerobot 83.6 s，mujoco 51 s，training extras 55.5 s，ffmpeg 42.1 s。合计约 12.5 分钟，pip 流量约 2.9 GB。最终环境体积 5358.7 MB，即 5.23 GB。

全过程没有任何 wheel 需要本地编译，不需要 MSVC。唯一在 PyPI 上没有 wheel 的包是 docopt 0.6.2，它是纯 Python，setuptools 就能装。

### 1.6 验证

```powershell
D:\cuda\envs\vlax\python.exe -c "import torch, numpy, mujoco, lerobot, transformers; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('numpy', numpy.__version__, 'mujoco', mujoco.__version__); import importlib.metadata as m; print('lerobot', m.version('lerobot'), 'transformers', m.version('transformers'))"
```

实测输出：

```
torch 2.11.0+cu126 True NVIDIA GeForce RTX 4060 Laptop GPU
numpy 2.2.6 mujoco 3.14.0
lerobot 0.6.1 transformers 5.5.4
```

### 1.7 torchcodec 的已知问题

在这个环境里 `import torchcodec` 会失败，报 `[WinError 127] 找不到指定的程序`，即 `libtorchcodec_core*.dll` 缺少导出。装了 conda-forge 的 ffmpeg 7.1.1 之后仍然失败，把 `Library\bin` 加到 PATH 也仍然失败，换成 bedbot 环境里那个能用的 torchcodec 0.11.0+cpu wheel 也一样失败。

原因是通道混用：bedbot 环境整体来自 conda-forge，而 vlax 的 base 来自 anaconda/pkgs/main，只有 ffmpeg 来自 conda-forge，DLL 依赖链断了。

lerobot 对此有容错，`get_safe_default_video_backend()` 捕获 ImportError、OSError、RuntimeError 并回退到 pyav。实测 pyav 编解码都正常：写出了 128x128 的 AV1 mp4，`decode_video_frames` 返回 `(3, 3, 64, 64) torch.float32`。

代价是每次 import lerobot 都会打印约 200 行 traceback，包括每个 spawn 出来的 dataloader worker 都打一遍。这会把真正的报错埋掉，实测时必须把输出重定向到文件再 grep。抑制办法是显式传 `--dataset.video_backend=pyav`，或者从一开始就用 `-c conda-forge` 建环境。后者没有实测，因为要重下 2.6 GB 的 torch。

---

## 2. 权重获取

### 2.1 需要两个仓库，不是一个

`lerobot/smolvla_base` 是策略 checkpoint。`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 是它的 VLM backbone。后者的仓库名不是硬编码，而是 config 字段 `vlm_model_name`。

加载策略时两个都会拉。读源码确认的顺序是：`PreTrainedPolicy.from_pretrained` 先构造 config，再调 `cls(config)`，`SmolVLAPolicy.__init__` 在这一步就构造 `SmolVLMWithExpertModel`，它调用 `AutoModelForImageTextToText.from_pretrained(model_id)`（因为发布的 config 里 `load_vlm_weights` 是 true）和 `AutoProcessor.from_pretrained(model_id)`，之后才轮到策略自己的 `model.safetensors`。

有意思的一点是 checkpoint 里本来就带了 VLM 权重：500 个张量中 345 个在 `model.vlm_with_expert.vlm.*` 下面，共 700,330,368 字节。也就是 backbone 的 2.03 GB 先被加载再被覆盖。但它不能跳过，因为 tokenizer 和 image processor 的配置只在 backbone 仓库里。

### 2.2 体积

实测的文件清单与字节数：

`lerobot/smolvla_base` 共 10 个文件 914,740,972 字节。真正需要的是 `*.json` 加 `*.safetensors`，共 906,718,631 字节。其中 `model.safetensors` 906,712,520 字节，`config.json` 2,299，`policy_preprocessor.json` 1,872，`policy_postprocessor.json` 660，两个 normalizer safetensors 各 640。剩下的是 README、一个 notebook 和 8 MB 的 `collage_small.gif`。

`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 共 37 个文件 7,917,649,732 字节，其中 5.88 GB 是 `onnx/` 目录，transformers 一次都不会碰。排除 onnx 后剩 13 个文件 2,034,846,684 字节，主要是 `model.safetensors` 2,029,990,624 加 `tokenizer.json` 3,548,256。

两边合计约 2.94 GB。务必排除 onnx，否则要多下 5.88 GB 的死文件。

### 2.3 下载命令

推荐落到一个固定的本地目录，理由见第 5 节的 Windows repo id bug。

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:HF_HOME="D:\hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"

D:\cuda\envs\vlax\Scripts\hf.exe download lerobot/smolvla_base --include "*.json" --include "*.safetensors" --local-dir D:/vla/ckpt/smolvla_base

D:\cuda\envs\vlax\Scripts\hf.exe download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --exclude "onnx/*" --local-dir D:/vla/ckpt/SmolVLM2-500M-Video-Instruct
```

实测速度：policy 906,721,522 字节 31.9 s，即 28.4 MB/s。backbone 2,034,857,549 字节 29.9 s，即 68.0 MB/s。用 python 的 `snapshot_download` 走同一个 endpoint 实测分别是 33.0 MB/s 和 68.8 MB/s。

关于 `HF_ENDPOINT` 有一处实测结果互相矛盾，这里如实写出。用 `snapshot_download` 和 `hf download` 整文件下载时，hf-mirror.com 给出上面那些几十 MB/s 的速度。但另一次针对同一台主机做的 10 MB ranged 测试只有 0.47 MB/s，而同一个 865 MB 文件直连 huggingface.co 走本机已有代理 127.0.0.1:7890 时达到 8.96 MB/s。两条路都被验证可用。建议先设 `HF_ENDPOINT` 试，如果几十秒内没有明显进度就 unset 走代理。

`--include` 和 `--exclude` 每个 flag 只接一个 pattern。写成 `--include "*.json" "*.safetensors"` 会把第二个吞成位置参数 FILENAME，实测报 `UserWarning: Ignoring --include since filenames have been explicitly set.` 然后 `OSError: [WinError 123]`，锁文件名成了 `*.safetensors.lock`。要重复写 flag。

### 2.4 两个环境变量的作用范围

`HF_ENDPOINT` 必须在下载进程内设好。huggingface_hub 在 import 时就把它读进 `huggingface_hub.constants.ENDPOINT`，import 之后再设无效。

`HF_HOME` 不只是模型缓存。读源码确认 `HF_LEROBOT_HOME = Path(HF_HOME)/'lerobot'`，`HF_LEROBOT_HUB_CACHE = HF_LEROBOT_HOME/'hub'`。所以设 `HF_HOME=D:\hf` 会同时把权重放到 `D:\hf\hub`，把录制和下载的 LeRobot 数据集放到 `D:\hf\lerobot`。录数据之前先把它指到大盘上。

默认值是 `%USERPROFILE%\.cache\huggingface`，缓存实际占用 2.74 GB，即 `models--lerobot--smolvla_base` 864.7 MB 加 `models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct` 1940.6 MB。

### 2.5 离线使用

权重进了缓存之后完全不需要网络。实测在 `HF_HUB_OFFLINE=1` 且 unset `HF_ENDPOINT` 的情况下，`snapshot_download` 取 policy 用 0.03 s，取 backbone 用 0.00 s，lerobot 实际调用的 `hf_hub_download('lerobot/smolvla_base','model.safetensors')` 立即返回 906,712,520 字节的文件。

`HF_HUB_OFFLINE` 是 huggingface_hub 的功能，lerobot 0.6.1 自己没有离线开关，全文 grep 也没有一处读这个变量。

离线时有两个坑。缓存里缺文件会抛 `huggingface_hub.errors.LocalEntryNotFoundError`。调 `snapshot_download` 时如果不带下载时用过的同一组 allow/ignore patterns，会抛 `IncompleteSnapshotError`，列出你当初故意跳过的 25 个文件（24 个 onnx 加 README）。用 `--local-dir` 落一个普通目录再把目录路径交给 lerobot 最省事。

指向本地目录是被验证过的路径。`from_pretrained` 里有 `if os.path.isdir(model_id)` 分支，直接读 `<dir>/model.safetensors`，config 从 `<dir>/config.json` 来。命令行上对应 `--policy.path=<dir>`。backbone 也能改到本地目录，办法是编辑 checkpoint 的 `config.json` 里的 `vlm_model_name`，或者命令行传 `--policy.vlm_model_name=<dir>`，两种都实测走通了 config 层。

### 2.6 一个路径长度限制

这台机器 `LongPathsEnabled` 是 0。`hf download --local-dir` 会把每个文件先落到 `<local-dir>\.cache\huggingface\download\<28字符id>.<64位hex>.<8位hex>.incomplete`，后缀约 161 个字符。实测一个 118 字符深的目标目录产生了 260 字符的暂存路径，报 `FileNotFoundError: [Errno 2]`，完全看不出是路径长度问题。同一条命令换到 62 字符的目标目录就成功了。

所以 checkpoint 目录要控制在大约 98 字符以内，`D:\vla\ckpt\smolvla_base` 这种就很安全。

---

## 3. 最小推理验证

### 3.1 脚本

核心部分如下，这是最小可用形态。

```python
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors   # 注意是 policies.factory

CKPT = "D:/vla/ckpt/smolvla_base"
DEV = "cuda"
CAM = "observation.images.camera1"   # 预训练 checkpoint 自己声明的相机名

policy = SmolVLAPolicy.from_pretrained(CKPT)
policy.to(DEV)
policy.eval()
policy.reset()

pre, post = make_pre_post_processors(
    policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}},
)

obs = {
    CAM: torch.rand(1, 3, 128, 128, dtype=torch.float32),   # 必须 float32 且在 [0,1]
    "observation.state": torch.rand(1, 6, dtype=torch.float32),
    "task": ["push the door open"],
}

with torch.inference_mode():
    action = post(policy.select_action(pre(obs)))      # (1, 6)，在 cpu 上
    chunk = policy.predict_action_chunk(pre(obs))      # (1, 50, 6)
```

必须走 `pre` 再进模型。直接把带 `task` 字符串的 batch 喂进 `select_action` 会报 `KeyError: observation.language.tokens`，因为分词这一步在 preprocessor 里。

preprocessor 做完之后 batch 的实际内容，实测：

```
observation.images.camera1            (1,3,128,128) float32 cuda:0 [0.000,1.000]
observation.language.tokens           (1,48)        int64   cuda:0
observation.language.attention_mask   (1,48)        bool    cuda:0
observation.state                     (1,6)         float32 cuda:0
task                                  ['push the door open\n']
```

注意最后那个 `\n`，是 `NewLineTaskProcessorStep` 自动加的。48 个位置里只有 5 个是真 token，其余是 padding，因为发布的 checkpoint 里 `pad_language_to` 是 `max_length`。

128x128 的图在 `prepare_images` 内部被放大到 512x512，因为是正方形所以没有黑边。

### 3.2 实测数字

加载时间，权重已在本地缓存，三次运行：

| 阶段 | 耗时 |
| --- | --- |
| `from_pretrained` | 7.67 / 7.74 / 7.99 s |
| `.to('cuda')` | 0.00 s |
| `make_pre_post_processors` | 0.17 / 0.18 s |
| 合计 | 7.85 / 7.92 / 8.18 s |

`.to('cuda')` 是 0 s，因为发布的 config 里 `device` 已经是 cuda，权重直接落在显卡上。第一次加载含下载是 74.7 s。

权重 dtype 由 checkpoint 自己决定，不需要手动 cast：vlm 是 bfloat16，lm_expert 是 bfloat16，`state_proj` 是 float32。

单 chunk 延迟，1 路 128x128 相机、6 维 state、48 个语言位置、batch 1、每 chunk 10 次 flow matching 迭代，每次都先 `policy.reset()`，两轮各 25 次：

| 指标 | 第一轮 | 第二轮 |
| --- | --- | --- |
| mean | 270.3 ms | 282.7 ms |
| median | 268.7 ms | 279.2 ms |
| p95 | 284.2 ms | 307.7 ms |
| min | 253.5 ms | 262.9 ms |
| max | 285.1 ms | 314.8 ms |

从队列里取一个已算好的动作是 2.49 到 2.56 ms，这个数里绝大部分是脚本自己造 batch 的开销，不是策略。

显存，`max_memory_reserved` 在 25 次循环上测得 952.0 MiB，`max_memory_allocated` 916.1 MiB，两轮完全一致。只加载权重之后是 883.3 MiB allocated，906.0 MiB reserved。也就是推理本身在权重之上只多用约 33 MiB。这大约是这张卡的 12%，8 GB 对推理完全不构成问题，显存的压力只在训练。

控制率，连续调 100 次 `select_action`：

| `n_action_steps` | 100 次耗时 | 等效频率 | 重规划次数 | 30 fps 下的开环窗口 |
| --- | --- | --- | --- | --- |
| 50 | 0.78 / 0.80 s | 128.6 / 124.4 Hz | 2 | 1.67 s |
| 10 | 2.95 / 3.64 s | 33.9 / 27.4 Hz | 10 | 0.33 s |

这两行的正确读法是：策略每秒只能重规划约 3.5 次，可达到的闭环频率就是 3.5 Hz 乘以 `n_action_steps`。`n_action_steps=50` 那个 128 Hz 只是吞吐量，不代表反馈，因为 1.67 s 内相机数据完全没被用到。推门有接触，建议设 `n_action_steps=10`，重新观测间隔 0.33 s，同时 33.9 Hz 仍然高于 30 Hz 的 MuJoCo 控制回路。

### 3.3 微调后的 checkpoint 回放

用第 5 节练出来的 checkpoint，喂第 4 节那个数据集里的真实一帧。

| 指标 | SmolVLA | ACT |
| --- | --- | --- |
| checkpoint 加载 | 7.8 s | 0.8 s |
| `predict_action_chunk` (1,50,6) | 0.244 s | 0.008 s |
| 首次 `select_action` | 0.525 s | 0.193 s |
| 随后 5 次合计 | 0.012 s | 0.004 s |
| 峰值显存 | 916.2 MiB | 215.9 MiB |

SmolVLA 单步分摊是 4.9 ms。在 30 Hz 下配 `n_action_steps=50` 相当于每 1.67 s 付一次 0.244 s，占用率约 15%，实时性有余量。

有一点要明白：这个 checkpoint 只练了 30 步，输出动作 `[0.072, 0.0358, 0.007, -0.0439, -0.0542, 0.0006]` 对应的真值是 `[0.0, 0.3, 0.0, -0.0, 0.25, 0.15]`，完全没收敛。这个测试证明的是接口、形状、速度和显存，不是策略质量。

---

## 4. 数据集格式

### 4.1 契约

lerobot 用的是 LeRobotDataset。你要写的是三件东西：一个 features 字典、每帧一个 dict、每条轨迹结束时存盘。

下面是实际跑通的构造脚本。本仓库实际使用的录制器是 `data/record.py`，结构相同。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 30
H = W = 128

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["lx", "ly", "lz", "rx", "ry", "rz"],
    },
    "observation.images.cam": {
        "dtype": "video",                 # 或 "image"，两者都映射到 VISUAL
        "shape": (H, W, 3),               # 注意 HWC，不是 CHW
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["dlx", "dly", "dlz", "drx", "dry", "drz"],
    },
}

def main():
    ds = LeRobotDataset.create(
        repo_id="local/backpack_push",
        fps=FPS,
        features=FEATURES,
        root="D:/vla/data/backpack_ds",
        robot_type="backpack_dual_arm",
        use_videos=True,
    )
    for ep in range(N_EP):
        task = "push the door open"
        for t in range(N_FRAMES):
            ds.add_frame({
                "observation.state": state.astype("float32"),   # (6,)
                "observation.images.cam": frame_uint8,          # (H,W,3) uint8
                "action": action.astype("float32"),             # (6,)
                "task": task,                                   # 每帧都要给
            })
        ds.save_episode()
    ds.finalize()

if __name__ == "__main__":     # Windows 上必须有，编码走子进程
    main()
```

几条硬约束。

`task` 是每帧必须给的字段，不在 FEATURES 里声明，直接放进 `add_frame` 的 dict。不给会在分词那一步抛 `KeyError: 'task'`，给 None 会抛 ValueError。

图像在 `add_frame` 里是 uint8 的 HWC。读回来的时候 LeRobotDataset 会转成 float32 的 CHW 且落在 [0,1]，这正是 SmolVLA 要的。这一转换只在通过 LeRobotDataset 读取时发生，手写的 MuJoCo 推理循环不会自动帮你做，见第 7 节。

`save_episode()` 每条轨迹调一次，`finalize()` 全部录完调一次。`finalize()` 会写 `meta/stats.json`，这是训练时归一化统计量的唯一来源，漏掉它后果严重。

`if __name__ == "__main__":` 保护在 Windows 上是必需的，因为视频编码会 spawn 子进程。

### 4.2 实际产出

实测 5 条轨迹各 60 帧，共 300 帧，构建耗时 4.3 s，磁盘 0.18 MB。文件清单：

```
     29989  data\chunk-000\file-000.parquet
      2783  meta\info.json
     10664  meta\stats.json
      2113  meta\tasks.parquet
     46668  meta\episodes\chunk-000\file-000.parquet
    100570  videos\observation.images.cam\chunk-000\file-000.mp4
```

`meta/info.json` 里记录 `"video.codec": "av1"`，`"video.fps": 30`，`"video.video_backend": "pyav"`，`total_episodes 5`，`total_frames 300`，`total_tasks 2`。

Windows 上视频编码不需要额外配置，PyAV 加 SVT-AV1 直接写出 AV1 mp4。过程中会打印 `Svt[info]: SVT [version]: SVT-AV1 Encoder Lib v3.0.0`、`Svt[warn]: Preset M12 is mapped to M10` 和 `[mp4 @ ...] Starting second pass: moving the moov atom to the beginning of the file`，都无害。SVT 那行由 worker 进程发出，会和你自己的 stdout 交错，看起来像是打印中途崩了。

### 4.3 6 到 14 维动作怎么落进 32 维

读源码确认的规则如下。

state 方向：`pad_vector` 把向量右侧零填充到 `max_state_dim=32`，然后过 `Linear(32, 960)`。注意这个函数在 `truncate=False` 时不截断，`if vector.shape[-1] >= new_dim: return vector` 原样返回。所以超过 32 维会崩，实测 33 维的报错是 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x33 and 32x960)`。

action 方向：同样零填充到 `max_action_dim=32`，模型输出 (B,50,32)，再按 `config.action_feature.shape[0]` 切回真实宽度。超过 32 维不报错，静默只给你前 32 列。

对你的项目，两条 7 自由度臂是 14 维，state 和 action 都远低于 32，两边都安全，不需要改 `max_state_dim` 或 `max_action_dim`。保持 32 与预训练一致。

一个代价要知道：训练损失是在 32 列上求平均的（`modeling_smolvla.py:312` 切到 `max_action_dim` 而不是真实维度），所以 14 维动作时有 18 列恒为零的 padding 也参与了 loss 平均。这是读源码得出的，没有做过 32 与 14 的对照实验，所以"抬高或压低 `max_action_dim` 是否更好"这个问题未实测。

### 4.4 相机命名，这一条决定你要不要传 rename_map

发布的 `lerobot/smolvla_base` 的 config 声明了三路相机 `observation.images.camera1`、`camera2`、`camera3`，分辨率 256x256，state 6 维，action 6 维。

`make_policy` 的行为是：`output_features` 总是被数据集覆盖，`input_features` 只在为空时才被数据集填充。预训练 checkpoint 的 `input_features` 非空，所以它的相机名和 state 维度会保留。

由此你有两个选择。要么把 MuJoCo 的相机在数据集里就命名成 `observation.images.camera1` 和 `camera2`，要么保留自己的名字并在训练时传 `--rename_map`。两条都可行，第二条已实测（见第 5 节）。

只有 1 到 2 路相机时，声明里剩下的槽位会被静默丢弃，不报警，prefix 长度从预训练时的 3x64 变成 1x64 或 2x64。这与预训练时的 token 布局不一致，实际影响未实测。

state 维度这一点要特别小心。因为 `input_features` 不被覆盖，14 维 state 配 `smolvla_base` 时策略 config 里声明的仍然是 6。模型层面不会崩，因为 14 会被零填充到 32，但 normalizer 那层按 6 维声明处理 14 维张量的行为没有实测。真到 14 维时，建议在本地 checkpoint 目录里把 `config.json` 的 `observation.state` shape 改成 14，然后跑一遍第 3 节的 smoke 脚本确认形状和归一化都对。这个改法本次未实测。

---

## 5. 训练

### 5.1 可用的 SmolVLA 微调命令

下面是实际跑通的那条，用 sh 写，因为 `--rename_map` 的 JSON 引号在 PowerShell 里的写法没有实测。

```sh
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

D:/cuda/envs/vlax/python.exe -m lerobot.scripts.lerobot_train \
  --policy.path="D:/vla/ckpt/smolvla_base" \
  --dataset.repo_id=local/backpack_push \
  --dataset.root="D:/vla/data/backpack_ds" \
  --rename_map='{"observation.images.cam": "observation.images.camera1"}' \
  --batch_size=8 \
  --steps=300 \
  --num_workers=2 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.video_backend=pyav \
  --log_freq=50 \
  --save_checkpoint=false \
  --output_dir="D:/vla/runs/smolvla_first"
```

逐项说明。

`--policy.path` 必须是本地目录，不能写 `lerobot/smolvla_base`。这是 Windows 上的 bug，详见 7.1。这个参数和 `--policy.type` 互斥，且它不出现在 `--help` 里，因为 parser 在 draccus 之前就把它从 argv 里摘走了。

`--dataset.repo_id` 即使用纯本地数据集也是必填项，`DatasetConfig.repo_id` 没有默认值。省掉会报 `error: dataset: Missing required field(s) repo_id for DatasetConfig`。配合 `--dataset.root` 时它只是个标签，不会联网，再加 `HF_HUB_OFFLINE=1` 就完全确定。

`--rename_map` 方向是数据集或环境的 key 映射到策略的 key。它要求必须有预训练 checkpoint，否则报错。它会被写进 checkpoint 的 preprocessor 里持久化，实测推理时喂 `observation.images.cam`，batch 里出来的就是 `observation.images.camera1`。

`--batch_size=8` 是实测的甜点，见 5.3。

`--num_workers`：短跑用 0，长跑用 2。Windows 上 `dataloader_multiprocessing_context` 默认是 spawn，worker 启动有一次性停顿，实测首个记录步的 `data_s` 在不同运行里是 0.221 s 与 2.191 到 2.542 s，之后掉到 0.004 s 附近。只跑 10 到 30 步的话这个停顿就是全部开销。

`--save_checkpoint=false` 用于第一次试跑，理由见 7.2，Windows 上存 checkpoint 会让进程非零退出。正式训练要去掉它。

`--dataset.video_backend=pyav` 用来压掉 torchcodec 的那堵 traceback。

其他默认值，读源码得到：`batch_size 8`，`steps 100000`，`num_workers 4`，`save_freq 20000`，`env_eval_freq 20000`，`log_freq 200`。

### 5.2 第一次跑什么

建议先跑 30 步 batch 4 确认不报错，再跑 300 步 batch 8 看 loss 是否下降。实测这两次分别是 26 s 和 95 s 挂钟时间，很便宜。

300 步 batch 8 的实测 loss，数据集只有 300 帧所以这是 8 个 epoch：

```
step:50  loss:0.899 grdn:7.407 updt_s:0.228 data_s:0.221 smp/s:18 mem_gb:2.32
step:100 loss:0.454 grdn:4.725 updt_s:0.217 data_s:0.004 smp/s:36 mem_gb:2.32
step:150 loss:0.267 grdn:3.551 updt_s:0.222 data_s:0.004 smp/s:35 mem_gb:2.32
step:200 loss:0.198 grdn:2.938 updt_s:0.222 data_s:0.006 smp/s:35 mem_gb:2.31
step:250 loss:0.163 grdn:2.476 updt_s:0.222 data_s:0.004 smp/s:35 mem_gb:2.32
step:300 loss:0.165 grdn:2.212 updt_s:0.225 data_s:0.005 smp/s:35 mem_gb:2.32
```

loss 从 0.899 降到 0.165，训练循环本身 79 s，加载模型 16 s。`num_learnable_params=99880992`，`num_total_params=450046176`，和第 0 节的数字一致。

还会看到一行 `Auto-scaling LR scheduler: num_training_steps (300) < num_decay_steps (30000)`，短跑时正常。

### 5.3 batch size 实测

SmolVLA 微调，fp32 AdamW，1 路 128x128 相机，`chunk_size=50`。`peak VRAM` 就是日志里的 `mem_gb`，来源是 `torch.cuda.max_memory_allocated()`。

| batch | peak VRAM | updt_s | steps/s |
| --- | --- | --- | --- |
| 4 | 1.80 GB | 0.239 | 3.21 |
| 8 | 2.33 GB | 0.219 | 4.3 到 4.5 |
| 16 | 3.37 GB | 0.515 | 1.51 |
| 32 | 5.30 GB | 0.67 到 0.83 | 1.05 |
| 48 | 7.31 GB | 2.30 到 4.41 | 0.24 到 0.30 |

五个 batch size 全部 exit 0，任何一档都没有抛 `torch.OutOfMemoryError`。8 GB 对这个任务是够的。batch 8 是甜点，32 是实用上限，48 时显卡已经溢到共享系统内存，`updt_s` 从 0.67 s 变成 2.3 到 4.4 s，慢十倍但不报错。这一点要靠盯 `updt_s` 发现，不要等异常。

`--policy.use_amp=true` 在 SmolVLA 上不可用，见 7.3。fp32 batch 8 只要 2.33 GB，用不着 AMP。

### 5.4 ACT 本地基线

ACT 不需要预训练权重，也就不需要 `--rename_map`，它直接采用数据集自己的 key `observation.images.cam`。

```sh
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1

D:/cuda/envs/vlax/python.exe -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=local/backpack_push \
  --dataset.root="D:/vla/data/backpack_ds" \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --batch_size=8 \
  --steps=300 \
  --num_workers=0 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.video_backend=pyav \
  --log_freq=50 \
  --save_checkpoint=false \
  --output_dir="D:/vla/runs/act_first"
```

`--policy.chunk_size=50` 是为了和 SmolVLA 对齐才显式写出来，方便比较。

30 步 batch 4 的实测 loss 从 55.399 降到 8.442，挂钟 13 s，`num_learnable_params=51571590`，全部可训练。

ACT 的 batch size 实测：

| batch | peak VRAM | updt_s | steps/s | data_s |
| --- | --- | --- | --- | --- |
| 4 | 0.94 GB | 0.068 | 7.23 | 0.066 |
| 8 | 0.94 GB | 0.064 | 5.07 | 0.123 |
| 16 | 0.97 GB | 0.068 | 2.98 | 0.250 |
| 32 | 1.44 GB | 0.151 | 2.08 | 0.334 |
| 64 | 1.90 GB | 0.171 | 0.88 | 0.910 |

batch 8 以上 ACT 已经不是 GPU 受限了，`updt_s` 一直停在 0.07 s 而 `data_s` 涨到 0.91 s，瓶颈是 CPU 上的 PyAV 视频解码。SmolVLA 全程都是 GPU 受限。

### 5.5 batch 4 上的正面对比

| 项 | ACT | SmolVLA |
| --- | --- | --- |
| 单步 GPU 时间 | 0.068 s | 0.239 s |
| 峰值显存 | 0.94 GB | 1.80 GB |
| 参数量 | 51.6M 全可训练 | 450.0M，99.9M 可训练 |
| checkpoint 体积 | 590.5 MB | 1258.3 MB |
| 单次推理 chunk | 0.008 s | 0.244 s |

### 5.6 checkpoint 布局

目录结构是 `<output_dir>/checkpoints/<零填充的步数>/`，下面两个子目录。实测文件清单：

```
checkpoints\000030\pretrained_model\config.json                                   2698
checkpoints\000030\pretrained_model\model.safetensors                        906712520
checkpoints\000030\pretrained_model\policy_postprocessor.json                      691
checkpoints\000030\pretrained_model\policy_postprocessor_step_0_unnormalizer_processor.safetensors   6544
checkpoints\000030\pretrained_model\policy_preprocessor.json                      2059
checkpoints\000030\pretrained_model\policy_preprocessor_step_5_normalizer_processor.safetensors      6544
checkpoints\000030\pretrained_model\train_config.json                             7808
checkpoints\000030\training_state\optimizer_param_groups.json                     9327
checkpoints\000030\training_state\optimizer_state.safetensors                 412659164
checkpoints\000030\training_state\rng_state.safetensors                          15708
checkpoints\000030\training_state\scheduler_state.json                             249
checkpoints\000030\training_state\training_step.json                                67
```

共 1258.3 MB。`checkpoints\last` 那个符号链接在 Windows 上建不出来（7.2），不影响使用。

断点续训实测可用且是样本精确的：

```sh
D:/cuda/envs/vlax/python.exe -m lerobot.scripts.lerobot_train \
  --config_path="D:/vla/runs/smolvla_first/checkpoints/000030/pretrained_model/train_config.json" \
  --resume=true --steps=40 --num_workers=0
```

实测输出里有 `Resuming data order at epoch 0, sample 120`，然后接着跑 step 35 和 40，exit 0。

### 5.7 归一化统计量，这一条最容易出事

统计量不是训练时算的，是建数据集时写进 `<root>/meta/stats.json` 的，训练时读进来。

从 `--policy.path` 微调会用新数据集的统计量覆盖预训练的统计量。`--resume` 则保留 checkpoint 里存的那份。实测确认：数据集的 `observation.state.mean` 是 `[-0.0, 0.0, 0.0038, 0.0, 0.0, -0.0088]`，`action.mean` 是 `[0.0, 0.0, -0.0019, -0.0, -0.0, 0.0044]`，微调出来的 checkpoint 里两项完全一致，8 个 key 各带 10 个统计量（mean、std、min、max、q01、q10、q50、q90、q99、count）都在。

现在说要紧的：`lerobot/smolvla_base` 本身没有你能用的统计量。它的 normalizer 里只有三个 key，`so100.buffer.action`、`so100-red.buffer.action`、`so100-blue.buffer.action`，比如 `so100.buffer.action.mean` 是 `[1.5959, 119.9442, 109.7698, 56.7055, -27.4227, 12.0028]`。根本没有 `observation.state` 和 `action` 这两个纯 key。而缺 key 时归一化是静默直通，不抛异常也不告警。

实测把 flow matching 的噪声种子固定后，用预训练 pipeline 和用空统计量的新建 pipeline，输出动作逐位相同。也就是说直接拿 `smolvla_base` 在你自己的机器人上做推理，归一化根本没生效。

微调之后这个问题自动消失，因为 `lerobot-train` 会把数据集的统计量写进 checkpoint。微调之前如果要手推理，必须显式传统计量：

```python
stats = {
    "observation.state": {"mean": mu_s, "std": sd_s},
    "action":            {"mean": mu_a, "std": sd_a},
}
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"},
                            "normalizer_processor": {"stats": stats}},
    postprocessor_overrides={"unnormalizer_processor": {"stats": stats}},
)
```

实测这样传之后，mean 10 std 2 的设定下 state 12.0 归一化成 1.0，动作反归一化按 `x*0.5+1` 正确执行。

另外 `dataset.use_imagenet_stats` 默认 true，会用 ImageNet 的 mean/std 覆盖每路相机自己的图像统计量。对 SmolVLA 无影响，因为它的 VISUAL 归一化模式是 IDENTITY，但 ACT 和 diffusion 会受影响。

---

## 6. 评测接入

### 6.1 走 lerobot-eval 需要两套约定

这条路已经端到端实测通过，但要同时满足两个互相独立的命名约定。

第一套是 draccus 的 choice 注册。写一个包，里面对一个 EnvConfig dataclass 调 `@EnvConfig.register_subclass("backpack")`，然后用 `--env.discover_packages_path=<包名>` 加载，或者把发行包命名成 `lerobot_env_*` 让它自动发现。

第二套是 gym 包名。`EnvConfig.package_name` 硬编码成 `f"gym_{self.type}"`，`gym_id` 硬编码成 `f"{package_name}/{task}"`。所以 `--env.type=backpack` 会去找一个名字就叫 `gym_backpack` 的可导入包，并期望它注册了 id `gym_backpack/Backpack-v0`。这是两个不同的包。

观测 key 也是硬编码的。`preprocess_observation` 只认三个名字：`pixels`（dict 时展开成 `observation.images.<k>`，非 dict 时变 `observation.image`）、`agent_pos` 映射到 `observation.state`、`environment_state` 映射到 `observation.env_state`。图像必须是 channel last 的 uint8，源码里有 assert。所以两条臂两路相机的环境要返回：

```python
{"pixels": {"cam_left": uint8_HWC, "cam_right": uint8_HWC},
 "agent_pos": proprio_vector_float32}
```

实测跑通的命令：

```sh
export PYTHONPATH=<放两个包的父目录>
D:/cuda/envs/vlax/python.exe -m lerobot.scripts.lerobot_eval \
  --policy.path="D:/vla/runs/smolvla_first/checkpoints/000030/pretrained_model" \
  --env.type=backpack \
  --env.discover_packages_path=lerobot_env_backpack \
  --eval.n_episodes=2 \
  --eval.batch_size=2 \
  --eval.use_async_envs=false \
  --policy.device=cuda \
  --output_dir="D:/vla/runs/eval_backpack"
```

实测输出里先是 `gym id 'gym_backpack/Backpack-v0' not found, attempting to import 'gym_backpack'...`，然后 `{'avg_sum_reward': 0.0, 'pc_success': 100.0, 'n_episodes': 2, 'video_paths': [...]}` 和 `End of eval`。结果写进 `output_dir/eval_info.json`。同样这两个 flag 在训练里配 `--env_eval_freq` 也生效。

`--eval.use_async_envs=false` 是 Windows 上必须的，理由见 7.4。

### 6.2 为什么自己写循环更简单

对你这个项目，我倾向于自己写 eval 循环，理由有四条。

一是观测 key 被硬编码成 `pixels`、`agent_pos`、`environment_state`，你要为了迁就它改 MuJoCo 环境的返回格式。

二是 `EnvConfig.package_name` 逼你把 gym 包命名成 `gym_<type>`，两个包的命名耦合在一起，改任务名要动两处。

三是 `AsyncVectorEnv` 在 Windows 上不可用，每次都要记得关掉。

四是手机 IMU 这类外部输入接进 gym 的 step 里比接进自己的循环麻烦。

自己写的循环核心就是 20 行：

```python
policy = SmolVLAPolicy.from_pretrained(CKPT); policy.to("cuda"); policy.eval()
pre, post = make_pre_post_processors(
    policy.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"}})

for episode in range(N):
    policy.reset()                    # 每条轨迹开头必须调
    mj_reset()
    for t in range(T):
        obs = {
            "observation.images.camera1": frame_float32_chw_01,  # (1,3,H,W)
            "observation.state": state_float32,                  # (1,d)
            "task": ["push the door open"],
        }
        with torch.inference_mode():
            action = post(policy.select_action(pre(obs)))         # (1,d)，cpu
        mj_apply(action[0].numpy())
```

两条必须遵守的纪律：每条轨迹开头调 `policy.reset()` 清动作队列，以及图像要自己转成 float32 CHW 的 [0,1]，LeRobotDataset 帮你做的那步在这里没有。

`select_action` 内部有 assert 拒绝 RTC，real time chunking 只能走 `predict_action_chunk`。

顺带一提，lerobot 0.6.1 有个 `phone` extra，内容是 hebi-py、teleop、fastapi、scipy，从名字看正对肩部手机 IMU 遥操作。本次没有安装也没有测试，hebi-py 在 Windows 上是否有 cp312 wheel 未知。

---

## 7. 新手会踩的坑

下面每一条都是实际撞过的，附真实报错文本和修法。

### 7.1 `--policy.path=lerobot/smolvla_base` 在 Windows 上必然失败

所有教程都这么写，而在 Windows 上它报：

```
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'.
The name cannot start or end with '-' or '.' and the maximum length is 96: 'lerobot\smolvla_base'.
```

原因是 `configs\train.py:177` 执行 `self.policy.pretrained_path = Path(policy_path)`，`WindowsPath('lerobot/smolvla_base')` 转字符串带反斜杠，`os.path.isdir()` 因此为 False，于是被当成 repo id 交给 `hf_hub_download`。

难认的地方在于 config 本身加载成功，你会先看到三百行 config dump 才崩，看起来像网络问题。它不是。

修法是把权重落成本地目录再传目录路径，见 2.3。`--reward_model.path` 有同样的 bug。

### 7.2 每次存 checkpoint 都让进程非零退出

```
OSError: [WinError 1314] <client does not hold the required privilege>: '000030' -> '...\checkpoints\last'
```

`common\train_utils.py:101` 调 `last_checkpoint_dir.symlink_to()`，Windows 上建符号链接需要 SeCreateSymbolicLinkPrivilege。SmolVLA 和 ACT 两条线都撞到了。

checkpoint 本身是完整的：它在 `lerobot_train.py:689` 写完，符号链接在 `:703` 才尝试。实测两个 checkpoint 都能正常加载做推理，也能从其中一个成功 resume。损失只是非零退出码和一个便利性的 `last` 链接。

三个修法：试跑时传 `--save_checkpoint=false`，实测干净 exit 0；或者接受这个崩溃，按显式步数引用 checkpoint；或者开 Windows 开发者模式或以管理员身份运行让符号链接被允许，这条没有实测，因为涉及改系统设置。

注意 `--save_freq=0` 躲不过去，因为最后一个 checkpoint 总会被写。

### 7.3 `--policy.use_amp=true` 在 SmolVLA 上直接崩

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BfBloat16'
```

真实报错里的类型名是 `'BFloat16'`，调用链是 `accelerate\accelerator.py:3047 clip_grad_norm_` 到 `torch\amp\grad_scaler.py:294`。原因是 SmolVLA 的 action expert 本来就是 bfloat16，torch 的 GradScaler 没法 unscale bf16 梯度。batch 8 和 32 都一样崩。

也就是说没有靠 AMP 换更大 batch 的路。影响不大，fp32 batch 8 只用 2.33 GB。

ACT 加 `--policy.use_amp=true` 不崩但没收益，`mem_gb` 还是 0.94，首个记录步的 `grdn` 是 nan，那是 scaler 在探测。

### 7.4 eval 默认走 forkserver，Windows 没有这个启动方式

```
ValueError: cannot find context for 'forkserver'
```

`envs\configs.py:115` 对 `AsyncVectorEnv` 硬编码 `extra_kwargs["context"] = "forkserver"`。而 `eval.batch_size` 默认是 0 表示自动（约 CPU 核数的 70%），`use_async_envs` 默认 true，所以默认配置必然走这条坏路。

修法是传 `--eval.use_async_envs=false`（实测可用），或者 `--eval.batch_size=1` 让它降级成 SyncVectorEnv。训练里的 env eval 同样要传。

### 7.5 微调时忘了 `--rename_map`

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'observation.images.camera2', 'observation.images.camera3']
- Extra features: ['observation.images.cam']
Please ensure your dataset and policy use consistent feature names.
If your dataset uses different observation keys (e.g., cameras named differently), use the `--rename_map` argument
```

这个报错很清楚甚至给了写法，照做即可。

但要知道两件事。传了 `--rename_map` 会同时关掉 `validate_visual_features_consistency`，所以一个拼写错误不再触发这个清晰的报错，而是变成下游一个莫名的错误。以及 `camera2` 和 `camera3` 之后是被静默丢弃的，没有任何提示。

### 7.6 缺归一化统计量是静默的

这一条最危险，因为不报错不告警。详见 5.7。守则是：推理前检查 checkpoint 的 normalizer 里有没有 `observation.state.*` 和 `action.*` 这两组纯 key，没有就说明归一化没生效。

只有一种缺失会报错，就是 checkpoint 目录里 normalizer 文件本身没拷过去：

```
FileNotFoundError: State file 'policy_preprocessor_step_5_normalizer_processor.safetensors'
was not found for local processor pipeline '...' at '...'
```

来自 `lerobot\processor\pipeline.py:1218`。

### 7.7 state 超过 32 维会崩，action 超过 32 维不会

33 维 state：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x33 and 32x960)
```

来自 `state_proj` 里的 `F.linear`。在 normalizer 之前还是之后注入这个 33 维向量，报错完全一样，因为 `pad_vector` 不截断。

40 维 action 则不报错，静默返回 (1,50,32)，你只拿到前 32 列。

两条臂 14 维两者都安全。

### 7.8 多给一路相机，静默丢掉

给 batch 里塞一个策略没声明的 `observation.images.wrist`，实测没有任何报错：`prepare_images` 只用了 1 张图，masks 是 `[[True]]`，动作照样是 (1,6)，那路相机凭空消失，prefix 悄悄变短了。

相机顺序也是隐式的，就是 `config.input_features` 的插入顺序。训练和部署时两路相机接反，不会有任何报错。

### 7.9 uint8 图像，两种不同的后果

这一点比"静默产生垃圾"更微妙，取决于分辨率。

128x128 的 uint8 图会崩，因为模型必须放大它：

```
NotImplementedError: "upsample_bilinear2d_out_frame" not implemented for 'Byte'
```

512x512 的 uint8 图不需要缩放，于是静默通过。`prepare_images` 出来的图 dtype 是 float32 但取值范围是 [-1.0, 507.0]，因为 `img*2-1` 作用在 0 到 255 上，动作是垃圾。

你的项目用 128x128，所以会得到那个崩溃，反而是好事。规则很简单：进模型的图必须是 float32 且在 [0,1]。LeRobotDataset 会帮你转，手写的 MuJoCo 循环不会。

### 7.10 import 路径

```
ImportError: cannot import name 'make_pre_post_processors' from 'lerobot.processor.factory'
Did you mean: 'make_default_processors'?
```

正确写法是 `from lerobot.policies.factory import make_pre_post_processors`。两个模块都存在且都有名字相近的 factory，很容易写错。签名是 `(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs)`。

### 7.11 直接给 `select_action` 传 `task` 字符串

```
KeyError: observation.language.tokens
```

分词在 preprocessor 里。必须先 `pre(obs)` 再进模型。

### 7.12 `--dataset.repo_id` 是必填项

```
error: `dataset`: Missing required field(s) `repo_id` for DatasetConfig
```

哪怕是纯本地数据集也要给。用 `local/backpack_push` 这种假名即可，配合 `--dataset.root` 不会联网。

### 7.13 空指令不报错

`task=""` 完全不报错。`NewLineTaskProcessorStep` 会把它变成 `"\n"`，分词成一个 token（id 198），模型靠图像和 state token 照样跑出 (1,6)。也就是说指令写错或者忘了写，你不会得到任何提示。缺 `task` 这个 key 才会抛 `KeyError: 'task'`。

### 7.14 显存溢出不报 OOM

batch 48 时 `mem_gb` 达到 7.31 GB，超过这张卡的 8188 MiB 之后 Windows WDDM 会溢到共享系统内存，`updt_s` 从 0.67 s 变成 2.3 到 4.4 s，慢十倍而进程 exit 0。别指望异常，盯 `updt_s`。

### 7.15 `pad_language_to` 的默认值和发布值不一样

dataclass 默认是 `"longest"`，而发布的 checkpoint 是 `"max_length"`，也就是固定 48 个位置。用 `"longest"` 时 prompt 长度会随 batch 变化，整个 prefix 长度和 KV cache 都跟着变。训练和部署要保持一致。

### 7.16 torchcodec 的 traceback 墙

每次 import lerobot 都会打印约 200 行 traceback，包括每个 spawn 出来的 worker 各打一遍。纯粹是噪声但会埋掉真正的报错。传 `--dataset.video_backend=pyav` 抑制。详见 1.7。

---

## 附录 A：本次未验证的事项

按第 7 条原则如实列出。

预训练权重在 `lerobot/smolvla_base` 上做端到端评测时是否产出合理的任务动作，没有测。所有推理测试用的是合成张量或未收敛的 30 步 checkpoint，只验证了管线、形状、速度和显存。

14 维 state 配 `smolvla_base` 的行为没有测。验证用的是 6 维。因为 `input_features` 不被数据集覆盖，14 维 state 时 normalizer 层的行为待确认，改 `config.json` 的 shape 这个修法也未实测。

只用 1 到 2 路相机时 prefix token 布局与预训练不一致带来的性能影响，没有测。显存和速度的测量里，第 3 节是 1 路相机，另一份独立测量用 3 路 512x512 在 batch 8 时是 3.6 GB，与本文 1 路的 2.33 GB 不能互换引用。

`load_vlm_weights=false` 能否跳过那 2.03 GB 的 backbone 权重，没有测。checkpoint 里确实带了 345 个 `vlm.*` 张量，看起来可以，但需要比对 missing/unexpected key 列表才能确认。

`max_action_dim` 从 32 抬高或压低的影响，没有测。已知 padding 的零列参与 loss 平均，但没有对照实验。

用 `-c conda-forge` 从头建环境能否修好 torchcodec，没有测，因为要重下 2.6 GB 的 torch。bedbot 环境里 torchcodec 可用，强烈提示答案是能。

`lerobot[phone]` extra 在 Windows 上能否装上，没有测。

PowerShell 里 `--rename_map` 的 JSON 引号写法，没有测。本文给的是 sh 形式。

开 Windows 开发者模式后 7.2 那个符号链接崩溃是否消失，没有测，因为涉及改系统设置。

`n_action_steps=10` 对接触阶段的实际效果，没有测。那个数字是按控制率推荐的，不是按任务成功率。
