import os
import time
import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.optimizer import Adam
from visualdl import LogWriter

try:
    from paddle.amp import GradScaler, auto_cast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from dataset import get_dataloaders, set_seed
from text_encoder import TextConditionEncoder, get_class_text_conditions, CIFAR10_CLASSES
from models import Generator, Discriminator

def train(args):
    set_seed(42)

    try:
        paddle.set_device('cpu')
        print("运行在 CPU 上")
    except Exception as e:
        print(f"iluvatar GPU 不可用，回退到 CPU: {e}")
        paddle.set_device('cpu')

    writer = LogWriter(logdir=args.log_dir)

    train_loader, val_loader, _ = get_dataloaders(args.data_root, batch_size=args.batch_size)

    text_enc = TextConditionEncoder(num_classes=10, cond_dim=128)
    class_conds = get_class_text_conditions(text_enc)
#优化器（学习率 2e-4、beta1=0.5）
    netG = Generator(z_dim=100, cond_dim=128)
    netD = Discriminator(cond_dim=128)

    if args.optimizer == 'adam':
        optG = Adam(parameters=netG.parameters(), learning_rate=args.lr_g, beta1=0.5, beta2=0.999)
        optD = Adam(parameters=netD.parameters(), learning_rate=args.lr_d, beta1=0.5, beta2=0.999)
    else:
        from paddle.optimizer import AdamW
        optG = AdamW(parameters=netG.parameters(), learning_rate=args.lr_g, beta1=0.5, beta2=0.999)
        optD = AdamW(parameters=netD.parameters(), learning_rate=args.lr_d, beta1=0.5, beta2=0.999)

    use_amp = args.amp and AMP_AVAILABLE
    if args.amp and not AMP_AVAILABLE:
        print("警告：AMP 不可用，回退到 FP32 训练")
    scaler = GradScaler() if use_amp else None

    # ---------- 断点续训 ----------
    start_epoch = 0
    if args.resume_epoch > 0:
        g_path = os.path.join(args.log_dir, f"G_epoch{args.resume_epoch}.pdparams")
        d_path = os.path.join(args.log_dir, f"D_epoch{args.resume_epoch}.pdparams")
        if os.path.exists(g_path) and os.path.exists(d_path):
            netG.set_state_dict(paddle.load(g_path))
            netD.set_state_dict(paddle.load(d_path))
            start_epoch = args.resume_epoch
            print(f"从 epoch {start_epoch} 继续训练")
        else:
            print(f"警告：未找到 epoch {args.resume_epoch} 的权重，从 0 开始")

    fixed_z = paddle.randn((10 * 10, 100))
    fixed_labels = paddle.arange(10).unsqueeze(1).tile((1, 10)).reshape((-1,))
    fixed_cond = text_enc(fixed_labels)

    total_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        netG.train()
        netD.train()
        d_losses, g_losses = [], []

        for i, (imgs, labels) in enumerate(train_loader):
            B = imgs.shape[0]
            cond = text_enc(labels)

            optD.clear_grad()
            if use_amp:
                with auto_cast(enable=True):
                    real_out = netD(imgs, cond)
                    d_real_loss = F.relu(1.0 - real_out).mean()
                    z = paddle.randn((B, 100))
                    fake_imgs = netG(z, cond)
                    fake_out = netD(fake_imgs.detach(), cond)
                    d_fake_loss = F.relu(1.0 + fake_out).mean()
                    d_loss = d_real_loss + d_fake_loss
                scaled = scaler.scale(d_loss)
                scaled.backward()
                scaler.minimize(optD, scaled)
            else:
                real_out = netD(imgs, cond)
                d_real_loss = F.relu(1.0 - real_out).mean()
                z = paddle.randn((B, 100))
                fake_imgs = netG(z, cond)
                fake_out = netD(fake_imgs.detach(), cond)
                d_fake_loss = F.relu(1.0 + fake_out).mean()
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optD.step()

            optG.clear_grad()
            if use_amp:
                with auto_cast(enable=True):
                    z = paddle.randn((B, 100))
                    fake_imgs = netG(z, cond)
                    fake_out = netD(fake_imgs, cond)
                    g_loss = -fake_out.mean()
                scaled_g = scaler.scale(g_loss)
                scaled_g.backward()
                scaler.minimize(optG, scaled_g)
            else:
                z = paddle.randn((B, 100))
                fake_imgs = netG(z, cond)
                fake_out = netD(fake_imgs, cond)
                g_loss = -fake_out.mean()
                g_loss.backward()
                optG.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        epoch_time = time.time() - epoch_start

        writer.add_scalar("Loss/D", sum(d_losses) / len(d_losses), epoch)
        writer.add_scalar("Loss/G", sum(g_losses) / len(g_losses), epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
#每 10 个 epoch 将生成样本转为网格图保存到 VisualDL，监控训练趋势
        if (epoch % 10) == 0:
            netG.eval()
            with paddle.no_grad():
                samples = netG(fixed_z, fixed_cond).detach()
            samples = (samples + 1) / 2
            try:
                import cv2
                imgs = samples[:64].numpy()
                imgs = np.transpose(imgs, (0, 2, 3, 1))
                imgs = (imgs * 255).astype(np.uint8)
                h, w = imgs.shape[1], imgs.shape[2]
                grid = np.zeros((h*8, w*8, 3), dtype=np.uint8)
                for idx in range(64):
                    row = idx // 8
                    col = idx % 8
                    grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = imgs[idx]
                grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                writer.add_image("Generated", grid_bgr, epoch)
            except Exception as e:
                print(f"图像记录失败: {e}")
            netG.train()
#实时记录训练硬件开销、打印训练状态、定期保存模型权重
        mem = 0.0
        try:
            if paddle.device.is_compiled_with_cuda():
                mem = paddle.device.cuda.max_memory_allocated() / 1024**3
            elif hasattr(paddle.device, 'custom_device_memory_stats'):
                stats = paddle.device.custom_device_memory_stats('iluvatar_gpu')
                mem = stats.get('allocated', 0) / 1024**3
        except Exception:
            pass
        if mem > 0:
            writer.add_scalar("Memory/VRAM_GB", mem, epoch)

        print(f"Epoch {epoch+1}/{args.epochs} | D_loss: {sum(d_losses)/len(d_losses):.4f} | "
              f"G_loss: {sum(g_losses)/len(g_losses):.4f} | Time: {epoch_time:.2f}s | VRAM: {mem:.2f}GB")

        if (epoch + 1) % 10 == 0:
            paddle.save(netG.state_dict(), os.path.join(args.log_dir, f"G_epoch{epoch+1}.pdparams"))
            paddle.save(netD.state_dict(), os.path.join(args.log_dir, f"D_epoch{epoch+1}.pdparams"))

    total_time = time.time() - total_start
    print(f"训练完成，总耗时: {total_time/3600:.2f}小时")
    writer.close()
#通过命令行灵活调整训练超参数、切换实验配置，并控制脚本的执行时机
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./cifar-10-batches-py')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='./logs/exp01')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='从指定 epoch 恢复训练，0 表示从头开始')
    args = parser.parse_args()
    train(args)