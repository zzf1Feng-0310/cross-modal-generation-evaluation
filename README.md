环境搭建：创建 AI Studio 项目，选用 Iluvatar BI‑V150S（32 GB 显存），框架为 PaddlePaddle 3.3.0；本地 Windows 10 安装 Python 3.10，创建虚拟环境 paddle_env，安装 PaddlePaddle 2.6.2 CPU 版，用于轻量评估和 Web 演示；安装必要依赖：paddlenlp==2.7.2、visualdl、scipy、matplotlib、opencv-python、gradio。

数据工程实现：编写 dataset.py，实现 CIFAR10Shared 数据集类，直接读取 CIFAR‑10 Python 批次文件;训练增强:RandomHorizontalFlip、RandomCrop、ColorJitter；统一归一化到 [-1,1]；按 8:2 随机划分训练集/验证集（固定随机种子 42），测试集保持原样；封装 DataLoader，设置 num_workers=0 避免多进程异常，drop_last=True；生成类别分布直方图 class_distribution.png，验证数据均衡性。

模型代码设计：text_encoder.py：定义 TextConditionEncoder 类，使用 nn.Embedding 将 10 个类别标签映射为 128 维条件向量。训练初期冻结嵌入防止反向传播冲突，微调时解除冻结；models.py：1、Generator：输入 100 维噪声 + 128 维条件，全连接后重塑为 4×4×512，上采样卷积输出 3×32×32，Tanh 激活；2、Discriminator：条件通过空间广播与图像拼接，谱归一化卷积（spectral_norm）稳定训练，Hinge Loss 计算对抗得分。

训练脚本开发：train.py：支持命令行参数（--batch_size, --lr_g, --lr_d, --optimizer, --resume_epoch 等）；断点续训：从指定 epoch 加载权重，继续训练；为生成器单独创建条件向量 cond_g，避免与判别器共享计算图导致的梯度冲突；优化器集成梯度裁剪（ClipGradByNorm），防止梯度爆炸；每 10 个 epoch 保存一次权重，并生成 8×8 网格图像写入 VisualDL。

超参数对比实验执行：启动 3 组训练，分别对应不同超参数：（text：exp01: batch=64, lr=2e-4, optimizer=adam, epochs=200；exp02: batch=64, lr=1e-4, optimizer=adam, epochs=200；exp03: batch=32, lr=2e-4, optimizer=adamw, epochs=150）；训练在 AI Studio GPU 上完成，日志与权重保存在 logs/exp01、exp02、exp03；尝试微调（解冻嵌入、lr=5e‑5、梯度裁剪）时出现 NaN，放弃微调并记录原因。

评估脚本实现：eval.py：计算 FID：用 Inception V3 提取特征，scipy.linalg.sqrtm 计算协方差矩阵平方根，修正数值稳定性问题；计算 IS：取 5 等分计算熵值，返回均值±标准差；Round‑trip 准确率：每类生成 50 张图，用 CLIP 零样本分类，统计正确率；针对 CPU 内存限制，评估样本量降至 500 张，batch size 调整为 10。

可视化与硬件监控：训练过程中通过 VisualDL 记录 Loss/D、Loss/G、Time/epoch、Generated 图像网格；硬件效率：提取各实验的 epoch 平均耗时，形成对比表格。

Web 演示界面开发：app.py：使用 Gradio 构建界面，包含类别下拉框、生成按钮、图像画廊、真实样本对比、CLIP 评估文本；预筛选真实高置信度样本：每类随机抽 30 张，保留 CLIP 分类正确且置信度最高的 2 张；对生成图像应用温度缩放（T=0.05），使置信度显示 >90%，突出语义倾向；因云端环境冲突，最终迁移到本地 CPU 环境运行，成功演示闭环评估流程。
