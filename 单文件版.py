# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
# from model import NeuralStyleTransferModel
# import settings
# import utils
import typing




#settings.py迁移
# 内容特征层及loss加权系数
CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
# 风格特征层及loss加权系数
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}
# 内容图片路径
#CONTENT_IMAGE_PATH = './images/content.jpg'
CONTENT_IMAGE_PATH = input("image path:")
# 风格图片路径
# STYLE_IMAGE_PATH = './images/style.jpg'
STYLE_IMAGE_PATH = input('style image path:')
# 生成图片的保存目录
# OUTPUT_DIR = './output'
OUTPUT_DIR = input('output path:')

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 1
# 风格loss总加权系数
STYLE_LOSS_FACTOR = 100

# 图片宽度
WIDTH = 450
# 图片高度
HEIGHT = 300

# 训练epoch数
EPOCHS = 20
# 每个epoch训练多少次
STEPS_PER_EPOCH = 100
# 学习率
LEARNING_RATE = 0.03







#utils.py迁移
# 我们准备使用经典网络在imagenet数据集上的与训练权重，所以归一化时也要使用imagenet的平均值和标准差
print("utils")
image_mean = tf.constant([0.485, 0.456, 0.406])
image_std = tf.constant([0.299, 0.224, 0.225])


def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean) / image_std


def load_images(image_path, width=WIDTH, height=HEIGHT):
    """
    加载并处理图片
    :param image_path:　图片路径
    :param width: 图片宽度
    :param height: 图片长度
    :return:　一个张量
    """
    # 加载文件
    x = tf.io.read_file(image_path)
    # 解码图片
    x = tf.image.decode_jpeg(x, channels=3)
    # 修改图片大小
    x = tf.image.resize(x, [height, width])
    x = x / 255.
    # 归一化
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    # 返回结果
    return x


def save_image(image, filename):
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)








#model.py迁移
print("models.py")
def get_vgg19_model(layers):
    """
    创建并初始化vgg19模型
    :return:
    """
    # 加载imagenet上预训练的vgg19
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # 提取需要被用到的vgg的层的output
    outputs = [vgg.get_layer(layer).output for layer in layers]
    # 使用outputs创建新的模型
    model = tf.keras.Model([vgg.input, ], outputs)
    # 锁死参数，不进行训练
    model.trainable = False
    return model


class NeuralStyleTransferModel(tf.keras.Model):

    def __init__(self, content_layers: typing.Dict[str, float] = CONTENT_LAYERS,
                 style_layers: typing.Dict[str, float] = STYLE_LAYERS):
        super(NeuralStyleTransferModel, self).__init__()
        # 内容特征层字典 Dict[层名,加权系数]
        self.content_layers = content_layers
        # 风格特征层
        self.style_layers = style_layers
        # 提取需要用到的所有vgg层
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        # 创建layer_name到output索引的映射
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        # 创建并初始化vgg网络
        self.vgg = get_vgg19_model(layers)

    def call(self, inputs, training=None, mask=None):
        """
        前向传播
        :return
            typing.Dict[str,typing.List[outputs,加权系数]]
        """
        outputs = self.vgg(inputs)
        # 分离内容特征层和风格特征层的输出，方便后续计算 typing.List[outputs,加权系数]
        content_outputs = []
        for layer, factor in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        # 以字典的形式返回输出
        return {'content': content_outputs, 'style': style_outputs}








# 创建模型
model = NeuralStyleTransferModel()

print("进入主程序")

# 加载内容图片
content_image = load_images(CONTENT_IMAGE_PATH)
# 风格图片
style_image = load_images(STYLE_IMAGE_PATH)

# 计算出目标内容图片的内容特征备用
target_content_features = model([content_image, ])['content']
# 计算目标风格图片的风格特征
target_style_features = model([style_image, ])['style']

M = WIDTH * HEIGHT
N = 3


def _compute_content_loss(noise_features, target_features):
    """
    计算指定层上两个特征之间的内容loss
    :param noise_features: 噪声图片在指定层的特征
    :param target_features: 内容图片在指定层的特征
    """
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    # 计算系数
    x = 2. * M * N
    content_loss = content_loss / x
    return content_loss


def compute_content_loss(noise_content_features):
    """
    计算并当前图片的内容loss
    :param noise_content_features: 噪声图片的内容特征
    """
    # 初始化内容损失
    content_losses = []
    # 加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    # 先交换维度，把channel维度提到最前面
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape，压缩成2d
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)


def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格loss
    :param noise_feature: 噪声图片在指定层的特征
    :param target_feature: 风格图片在指定层的特征
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    # 计算系数
    x = 4. * (M ** 2) * (N ** 2)
    return style_loss / x


def compute_style_loss(noise_style_features):
    """
    计算并返回图片的风格loss
    :param noise_style_features: 噪声图片的风格特征
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)


def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features['content'])
    style_loss = compute_style_loss(noise_features['style'])
    return content_loss * CONTENT_LOSS_FACTOR + style_loss * STYLE_LOSS_FACTOR


# 使用Adma优化器
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# 基于内容图片随机生成一张噪声图片
noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, HEIGHT, WIDTH, 3))) / 2)


# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss


# 创建保存生成图片的文件夹
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# 共训练EPOCHS个epochs
for epoch in range(EPOCHS):
    # 使用tqdm提示训练进度
    with tqdm(total=STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, EPOCHS)) as pbar:
        # 每个epoch训练STEPS_PER_EPOCH次
        for step in range(STEPS_PER_EPOCH):
            _loss = train_one_step()
            pbar.set_postfix({'loss': '%.4f' % float(_loss)})
            pbar.update(1)
        # 每个epoch保存一次图片
        save_image(noise_image, '{}/{}.jpg'.format(OUTPUT_DIR, epoch + 1))
