# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2
import argparse
import typing
import h5py

# 解析命令行参数
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_path", type=str, default="./images/1.jpg", help="原图路径")
    parser.add_argument("--style_img_path", type=str, default="./images/style.jpg", help="风格图片路径")
    parser.add_argument("--output_path", type=str, default="./output/1", help="生成图片保存路径")
    parser.add_argument("--epochs", type=int, default=20, help="总训练轮数")
    parser.add_argument("--step_per_epoch", type=int, default=100, help="每轮训练次数")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--content_loss_factor", type=float, default=1.0, help="内容损失总加权系数")
    parser.add.argument("--style_loss_factor", type=float, default=100.0, help="风格损失总加权系数")
    parser.add.argument("--img_size", type=int, default=0, help="图片尺寸，0代表不设置使用默认尺寸(450*300)，输入1代表使用图片尺寸，其他输入代表使用自定义尺寸")
    parser.add.argument("--img_width", type=int, default=450, help="自定义图片宽度")
    parser.add.argument("--img_height", type=int, default=300, help="自定义图片高度")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def load_images(image_path, width, height):
    """
    加载并处理图片，返回一个张量
    """
    x = tf.io.read_file(image_path)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [height, width])
    x = x / 255.0
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    return x

def load_images_from_list(image_array, width, height):
    """
    从numpy数组加载并处理图片，返回一个张量
    """
    x = tf.convert_to_tensor(image_array, dtype=tf.float32)
    x = tf.image.resize(x, [height, width])
    x = x / 255.0
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    return x

def save_image(image, filename):
    """
    保存图片
    """
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.0
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)

def save_image_for_gradio(image):
    """
    将图片保存为numpy数组
    """
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.0
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    numpy_array = x.numpy()  # 将TensorFlow张量转换为numpy数组
    return numpy_array

def get_vgg19_model(layers):
    """
    创建并初始化vgg19模型
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    outputs = [vgg.get_layer(layer).output for layer in layers]
    model = tf.keras.Model(vgg.input, outputs)
    model.trainable = False
    return model

class NeuralStyleTransferModel(tf.keras.Model):
    def __init__(self, content_layers: typing.Dict[str, float], style_layers: typing.Dict[str, float]):
        super(NeuralStyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        self.vgg = get_vgg19_model(layers)

    def call(self, inputs, training=None, mask=None):
        outputs = self.vgg(inputs)
        content_outputs = []
        for layer, factor in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        return {"content": content_outputs, "style": style_outputs}

def normalization(x):
    """
    对输入图片进行归一化处理，返回归一化后的值
    """
    return (x - image_mean) / image_std

def _compute_content_loss(noise_features, target_features):
    """
    计算指定层上两个特征之间的内容损失
    """
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    x = 2.0 * M * N
    content_loss = content_loss / x
    return content_loss

def compute_content_loss(noise_content_features, target_content_features):
    """
    计算并返回当前图片的内容损失
    """
    content_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)

def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    x = tf.transpose(feature, perm=[2, 0, 1])
    x = tf.reshape(x, (x.shape[0], -1))
    return x @ tf.transpose(x)

def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格损失
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    x = 4.0 * (M**2) * (N**2)
    return style_loss / x

def compute_style_loss(noise_style_features, target_style_features):
    """
    计算并返回图片的风格损失
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)

def total_loss(noise_features, target_content_features, target_style_features):
    """
    计算总损失
    """
    content_loss = compute_content_loss(noise_features["content"], target_content_features)
    style_loss = compute_style_loss(noise_features["style"], target_style_features)
    return content_loss * CONTENT_LOSS_FACTOR + style_loss * STYLE_LOSS_FACTOR

@tf.function
def train_one_step(model, noise_image, optimizer, target_content_features, target_style_features):
    """
    一次迭代过程
    """
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs, target_content_features, target_style_features)
    grad = tape.gradient(loss, noise_image)
    optimizer.apply_gradients([(grad, noise_image)])
    return loss

def main(content_img, style_img, epochs, step_per_epoch, learning_rate, content_loss_factor, style_loss_factor, img_size, img_width, img_height, progress_callback=None):
    global CONTENT_LOSS_FACTOR, STYLE_LOSS_FACTOR, CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH, OUTPUT_DIR, EPOCHS, LEARNING_RATE, STEPS_PER_EPOCH, M, N, image_mean, image_std, IMG_WIDTH, IMG_HEIGHT

    CONTENT_LOSS_FACTOR = content_loss_factor
    STYLE_LOSS_FACTOR = style_loss_factor
    CONTENT_IMAGE_PATH = content_img
    STYLE_IMAGE_PATH = style_img
    EPOCHS = epochs
    LEARNING_RATE = learning_rate
    STEPS_PER_EPOCH = step_per_epoch

    # 内容特征层及损失加权系数
    CONTENT_LAYERS = {"block4_conv2": 0.5, "block5_conv2": 0.5}
    # 风格特征层及损失加权系数
    STYLE_LAYERS = {
        "block1_conv1": 0.2,
        "block2_conv1": 0.2,
        "block3_conv1": 0.2,
        "block4_conv1": 0.2,
        "block5_conv1": 0.2,
    }

    if img_size == "default size":
        IMG_WIDTH = 450
        IMG_HEIGHT = 300
    else:
        IMG_WIDTH = img_width
        IMG_HEIGHT = img_height

    print("IMG_WIDTH:", IMG_WIDTH)
    print("IMG_HEIGHT:", IMG_HEIGHT)

    # 我们准备使用经典网络在imagenet数据集上的预训练权重，所以归一化时也要使用imagenet的平均值和标准差
    image_mean = tf.constant([0.485, 0.456, 0.406])
    image_std = tf.constant([0.299, 0.224, 0.225])

    model = NeuralStyleTransferModel(CONTENT_LAYERS, STYLE_LAYERS)

    content_image = load_images_from_list(CONTENT_IMAGE_PATH, IMG_WIDTH, IMG_HEIGHT)
    style_image = load_images_from_list(STYLE_IMAGE_PATH, IMG_WIDTH, IMG_HEIGHT)

    target_content_features = model(content_image)["content"]
    target_style_features = model(style_image)["style"]

    M = IMG_WIDTH * IMG_HEIGHT
    N = 3

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    noise_image = tf.Variable((content_image[0] + np.random.uniform(-0.2, 0.2, (1, IMG_HEIGHT, IMG_WIDTH, 3))) / 2)

    total_steps = EPOCHS * STEPS_PER_EPOCH
    current_step = 0

    for epoch in range(EPOCHS):
        with tqdm(total=STEPS_PER_EPOCH, desc="Epoch {}/{}".format(epoch + 1, EPOCHS)) as pbar:
            for step in range(STEPS_PER_EPOCH):
                _loss = train_one_step(model, noise_image, optimizer, target_content_features, target_style_features)
                pbar.set_postfix({"loss": "%.4f" % float(_loss)})
                pbar.update(1)

                current_step += 1
                if progress_callback:
                    progress_callback(current_step / total_steps)
    
    return save_image_for_gradio(noise_image)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt.content_img_path, opt.style_img_path, opt.epochs, opt.step_per_epoch, opt.learning_rate, opt.content_loss_factor, opt.style_loss_factor, opt.img_size, opt.img_width, opt.img_height)
