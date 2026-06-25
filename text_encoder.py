import paddle
import paddle.nn as nn

#定义 CIFAR10 类别列表，和数据集标签一一对应，是文本条件的基础
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

class TextConditionEncoder(nn.Layer):
    """使用可学习嵌入替代 BERT，嵌入固定不训练（避免反向传播冲突）"""
    def __init__(self, num_classes=10, cond_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, cond_dim)
        # 冻结嵌入，不求梯度，避免后续两次 backward 冲突
        self.embedding.weight.stop_gradient = True

    def forward(self, labels):
        return self.embedding(labels)

#预生成所有类别的文本嵌入，返回固定向量供训练 / 评估直接调用，提升效率。
def get_class_text_conditions(encoder):
    all_labels = paddle.arange(10)
    with paddle.no_grad():
        conds = encoder(all_labels)
    return conds