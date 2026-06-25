import os, gc
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
import paddle
from PIL import Image

from dataset import get_dataloaders, set_seed
from text_encoder import TextConditionEncoder, get_class_text_conditions, CIFAR10_CLASSES
from models import Generator

# ====== FID 计算（修正协方差矩阵平方根取迹） ======
def compute_fid(real_loader, fake_images, batch_size=10):
    inception = paddle.vision.models.inception_v3(pretrained=True)#加载预训练 InceptionV3
    inception.fc = paddle.nn.Identity()
    inception.eval()
#计算真实 / 生成图像的特征均值、协方差，修正协方差矩阵平方根的复数问题，FID 越小说明生成图像越接近真实；
    def get_feats_from_loader(loader, max_samples=500):
        feats = []
        processed = 0
        for imgs, _ in loader:
            if processed >= max_samples:
                break
            imgs = paddle.nn.functional.interpolate(imgs, size=(299, 299), mode='bilinear')
            feat = inception(imgs).squeeze(-1).squeeze(-1)
            feats.append(feat.numpy())
            processed += imgs.shape[0]
            del imgs, feat
        return np.concatenate(feats, axis=0)

    def get_feats_from_tensor(tensor, max_samples=500, bs=10):
        feats = []
        tensor = tensor[:max_samples]
        for i in range(0, len(tensor), bs):
            batch = tensor[i:i+bs]
            batch = paddle.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear')
            feat = inception(batch).squeeze(-1).squeeze(-1)
            feats.append(feat.numpy())
            del batch, feat
        return np.concatenate(feats, axis=0)

    real_feats = get_feats_from_loader(real_loader, max_samples=500)
    fake_feats = get_feats_from_tensor(fake_images, max_samples=500, bs=batch_size)

    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2

    # 矩阵平方根并取迹
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    del real_feats, fake_feats, mu1, sigma1, mu2, sigma2, diff, covmean
    gc.collect()
    return float(fid)

# 用 InceptionV3 预测生成图像类别概率，计算 IS 值
def compute_is(fake_images, n_split=5):
    inception = paddle.vision.models.inception_v3(pretrained=True)
    inception.eval()
    fake_299 = paddle.nn.functional.interpolate(fake_images, size=(299, 299), mode='bilinear')
    preds = inception(fake_299)
    preds = paddle.nn.functional.softmax(preds, axis=1).numpy()
    del fake_299, inception
    gc.collect()

    scores = []
    for k in range(n_split):
        part = preds[k::n_split]
        py = np.mean(part, axis=0, keepdims=True)
        scores.append(np.exp(np.mean([entropy(p, py[0]) for p in part])))
    return np.mean(scores), np.std(scores)

# 将生成图像反归一化到 [0,255]，用 CLIP 实现零样本分类，返回预测类别
def clip_zero_shot(fake_images, clip_model, clip_processor, class_texts):
    np_imgs = []
    for t in fake_images:
        # 反归一化到 [0,1]，再转为 uint8 HWC 格式
        img = (t + 1) / 2
        arr = (img * 255).clip(0, 255).transpose((1, 2, 0)).astype('uint8')
        np_imgs.append(arr)
    inputs = clip_processor(images=np_imgs, text=class_texts, return_tensors='pd', padding=True)
    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image
    pred = paddle.argmax(logits, axis=1)
    return pred

# 每类生成 50 张图像，用 CLIP 预测类别，计算 “文本→图像→文本” 的闭环准确率
def round_trip_accuracy(generator, text_enc, clip_model, clip_processor, device):
    generator.eval()
    all_fakes, all_labels = [], []
    for class_id in range(10):
        labels_tensor = paddle.full((50,), class_id, dtype='int64')
        cond = text_enc(labels_tensor)
        z = paddle.randn((50, 100))
        fake = generator(z, cond)
        all_fakes.append(fake)
        all_labels.extend([class_id] * 50)
    all_fakes = paddle.concat(all_fakes, axis=0)
    texts = [f"a photo of a {cls}" for cls in CIFAR10_CLASSES]
    pred = clip_zero_shot(all_fakes, clip_model, clip_processor, texts)
    acc = (pred.numpy() == np.array(all_labels)).mean()
    return acc

if __name__ == "__main__":
    set_seed(42)

    # 设置设备（CPU）
    paddle.set_device('cpu')
    print("运行在 CPU 上")

    # 加载模型
    text_enc = TextConditionEncoder(num_classes=10, cond_dim=128)
    generator = Generator(100, 128)

    # ★ 修改此处路径，依次评估 exp01 / exp02 / exp03
    model_path = './logs/exp03/G_epoch150.pdparams'
    generator.set_state_dict(paddle.load(model_path))
    generator.eval()

    # 加载训练好的生成器，生成 500 张图像计算 FID/IS，加载 CLIP 模型计算 Round-trip 准确率，全面评估生成质量
    print("生成 500 张图像...")
    fake_500 = []
    for _ in range(5):
        labels = paddle.randint(0, 10, (100,))
        cond = text_enc(labels)
        z = paddle.randn((100, 100))
        fake_500.append(generator(z, cond))
    fake_500 = paddle.concat(fake_500, axis=0)

    # ---------- FID ----------
    _, val_loader, _ = get_dataloaders('./cifar-10-batches-py', batch_size=10)
    print("计算 FID（batch=10，样本=500）...")
    fid = compute_fid(val_loader, fake_500, batch_size=10)
    print(f"FID: {fid:.2f}")

    # ---------- IS ----------
    print("计算 IS...")
    is_mean, is_std = compute_is(fake_500, n_split=5)
    print(f"IS: {is_mean:.2f} ± {is_std:.2f}")

    del fake_500
    gc.collect()

    # ---------- Round‑trip 评估 ----------
    try:
        from paddlenlp.transformers import CLIPModel, CLIPProcessor
        print("加载 CLIP 模型...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("计算 Round‑trip 准确率（每类 50 张）...")
        rt_acc = round_trip_accuracy(generator, text_enc, clip_model, clip_processor, device='cpu')
        print(f"Round-trip accuracy: {rt_acc:.4f}")
    except Exception as e:
        print(f"CLIP 评估失败: {e}")