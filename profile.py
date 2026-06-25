import time
import paddle
from dataset import get_dataloaders
from text_encoder import TextConditionEncoder, get_class_text_conditions
from models import Generator, Discriminator

def run_profile(config):
    print(f"\n=== Profile: {config['name']} ===")
    # 设备设置
    try:
        paddle.set_device('cpu')
        print("运行在 CPU 上")
    except Exception as e:
        print(f"GPU 不可用，退出 Profile: {e}")
        return None, None

    # 文本编码器
    text_enc = TextConditionEncoder(cond_dim=128)
    class_conds = get_class_text_conditions(text_enc)

    # 数据加载
    train_loader, _, _ = get_dataloaders('./cifar-10-batches-py', batch_size=config['batch'])

    # 模型
    netG = Generator(100, 128)
    netD = Discriminator(128)

    # 优化器
    optG = paddle.optimizer.Adam(parameters=netG.parameters(), learning_rate=2e-4, beta1=0.5, beta2=0.999)
    optD = paddle.optimizer.Adam(parameters=netD.parameters(), learning_rate=2e-4, beta1=0.5, beta2=0.999)

    # 混合精度
    try:
        from paddle.amp import GradScaler, auto_cast
        AMP_OK = True
    except ImportError:
        AMP_OK = False
    use_amp = config['amp'] and AMP_OK
    scaler = GradScaler() if use_amp else None

    # 预热（避免冷启动影响性能数据）
    print("预热...")
    for i, (imgs, labels) in enumerate(train_loader):
        if i >= 5:
            break
        B = imgs.shape[0]
        cond = class_conds[labels]
        if use_amp:
            with auto_cast(enable=True):
                z = paddle.randn((B, 100))
                fake = netG(z, cond)
                loss = netD(fake, cond).mean()#清空显存统计，保证测试准确性
            scaler.scale(loss).backward()
        else:
            z = paddle.randn((B, 100))
            fake = netG(z, cond)
            loss = netD(fake, cond).mean()
            loss.backward()
        optD.clear_grad()
        optG.clear_grad()

    # 清空显存统计
    try:
        paddle.device.cuda.empty_cache()
        paddle.device.cuda.reset_peak_memory_stats()
    except:
        pass

    print("正式测试...")
    start = time.time()
    for i, (imgs, labels) in enumerate(train_loader):
        if i >= 10:          # 只测10个batch
            break
        B = imgs.shape[0]
        cond = class_conds[labels]
        optD.clear_grad()
        if use_amp:
            with auto_cast(enable=True):
                real_out = netD(imgs, cond)
                z = paddle.randn((B, 100))
                fake = netG(z, cond)
                fake_out = netD(fake.detach(), cond)
                d_loss = paddle.nn.functional.relu(1 - real_out).mean() + \
                         paddle.nn.functional.relu(1 + fake_out).mean()
            scaled = scaler.scale(d_loss)
            scaled.backward()
            scaler.minimize(optD, scaled)
        else:
            real_out = netD(imgs, cond)
            z = paddle.randn((B, 100))
            fake = netG(z, cond)
            fake_out = netD(fake.detach(), cond)
            d_loss = paddle.nn.functional.relu(1 - real_out).mean() + \
                     paddle.nn.functional.relu(1 + fake_out).mean()
            d_loss.backward()
            optD.step()

    # 同步并计算时间
    try:
        paddle.device.cuda.synchronize()
    except:
        pass
    elapsed = time.time() - start

    # 获取显存峰值
    mem = 0.0
    try:
        if paddle.device.is_compiled_with_cuda():
            mem = paddle.device.cuda.max_memory_allocated() / 1024**3
        elif hasattr(paddle.device, 'custom_device_memory_stats'):
            stats = paddle.device.custom_device_memory_stats('iluvatar_gpu')
            mem = stats.get('allocated', 0) / 1024**3
    except:
        pass

    print(f"10 batches 耗时: {elapsed:.2f}s, 显存峰值: {mem:.2f}GB")
    return elapsed, mem


if __name__ == "__main__":
    results = {}
    for cfg in [
        {"name": "FP32_batch32", "batch": 32, "amp": False},
        {"name": "AMP_batch64", "batch": 64, "amp": True}
    ]:
        t, mem = run_profile(cfg)
        if t is not None:
            results[cfg['name']] = (t, mem)

    print("\n===== Profile 结果汇总 =====")
    for k, v in results.items():
        print(f"{k}: 耗时={v[0]:.2f}s, 显存={v[1]:.2f}GB")