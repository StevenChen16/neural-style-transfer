# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import cv2
import argparse
import typing
import logging
import platform
import subprocess
from pathlib import Path

# 设置TensorFlow环境变量以启用详细日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 显示所有消息

# 尝试解决CUDA可见性问题
if platform.system() == 'Windows':
    # Windows平台特定的设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 显式设置使用第一个GPU
else:
    # Linux/MacOS平台特定的设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 打印TensorFlow版本和构建信息
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}" if hasattr(tf.test, 'is_gpu_available') else 
      f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# 尝试打印CUDA信息
try:
    if hasattr(tf.sysconfig, 'get_build_info'):
        cuda_version = tf.sysconfig.get_build_info().get('cuda_version', 'unknown')
        cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version', 'unknown')
        print(f"CUDA version: {cuda_version}")
        print(f"cuDNN version: {cudnn_version}")
except Exception as e:
    print(f"无法获取CUDA版本信息: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neural_style_transfer")

# 解析命令行参数
def parse_opt(known=False):
    parser = argparse.ArgumentParser(description="Neural Style Transfer Implementation")
    parser.add_argument("--content_img_path", type=str, default="./images/1.jpg", help="原图路径")
    parser.add_argument("--style_img_path", type=str, default="./images/style.jpg", help="风格图片路径")
    parser.add_argument("--output_path", type=str, default="./output/1", help="生成图片保存路径")
    parser.add_argument("--epochs", type=int, default=20, help="总训练轮数")
    parser.add_argument("--step_per_epoch", type=int, default=100, help="每轮训练次数")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    parser.add_argument("--content_loss_factor", type=float, default=1.0, help="内容损失总加权系数")
    parser.add_argument("--style_loss_factor", type=float, default=100.0, help="风格损失总加权系数")
    parser.add_argument("--img_size", type=int, default=0, 
                        help="图片尺寸，0代表使用默认尺寸(450*300)，1代表使用原图尺寸，其他输入代表使用自定义尺寸")
    parser.add_argument("--img_width", type=int, default=450, help="自定义图片宽度")
    parser.add_argument("--img_height", type=int, default=300, help="自定义图片高度")
    parser.add_argument("--device", type=str, default="", 
                        help="计算设备，可选项：'cpu', 'gpu', 'gpu:0', 'gpu:1'等，留空则自动选择最佳设备")
    parser.add_argument("--random_init", action="store_true", help="使用随机初始化而不是基于内容图片")
    parser.add_argument("--save_every", type=int, default=1, help="每隔多少个epoch保存一次图片")
    parser.add_argument("--output_format", type=str, default="jpg", choices=["jpg", "png"], help="输出图片格式")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

class NeuralStyleTransfer:
    """神经风格迁移实现类"""
    
    # 图像归一化参数 (ImageNet预训练模型标准)
    IMAGE_MEAN = tf.constant([0.485, 0.456, 0.406])
    IMAGE_STD = tf.constant([0.299, 0.224, 0.225])
    
    # 默认的内容特征层及损失加权系数
    DEFAULT_CONTENT_LAYERS = {"block4_conv2": 0.5, "block5_conv2": 0.5}
    
    # 默认的风格特征层及损失加权系数
    DEFAULT_STYLE_LAYERS = {
        "block1_conv1": 0.2,
        "block2_conv1": 0.2,
        "block3_conv1": 0.2,
        "block4_conv1": 0.2,
        "block5_conv1": 0.2,
    }
    
    def __init__(self, config):
        """初始化神经风格迁移模型
        
        Args:
            config: 包含所有配置参数的对象
        """
        self.config = config
        
        # 设置计算设备
        self._setup_device()
        
        # 设置图片尺寸
        self._setup_image_size()
        
        # 内容和风格特征层配置
        self.content_layers = self.DEFAULT_CONTENT_LAYERS
        self.style_layers = self.DEFAULT_STYLE_LAYERS
        
        # 创建目录
        self._create_output_dir()
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载图片
        self.content_image = self._load_image(config.content_img_path)
        self.style_image = self._load_image(config.style_img_path)
        
        # 计算目标特征
        self._compute_target_features()
        
        # 创建优化器
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        
        # 初始化生成图像
        self._init_generated_image()
        
    def _setup_device(self):
        """设置计算设备"""
        # 确保TensorFlow看到GPU
        physical_devices = tf.config.list_physical_devices()
        gpus = tf.config.list_physical_devices('GPU')
        
        logger.info(f"Available physical devices: {[d.name for d in physical_devices]}")
        logger.info(f"Available GPUs: {[g.name for g in gpus]}")
        
        # 配置GPU内存增长
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Enabled memory growth for GPU: {gpu.name}")
                except Exception as e:
                    logger.warning(f"Error setting memory growth: {e}")
        
        # 确定使用哪个设备
        if not self.config.device:
            # 自动选择可用设备
            if gpus:
                try:
                    # 设置可见GPU
                    self.device = "/device:GPU:0"
                    logger.info(f"Auto-selected GPU device")
                except RuntimeError as e:
                    logger.warning(f"GPU error: {e}")
                    self.device = "/device:CPU:0"
            else:
                logger.warning("No GPUs found, using CPU")
                self.device = "/device:CPU:0"
        else:
            # 使用用户指定的设备
            device = self.config.device.lower()
            if device == 'cpu':
                self.device = "/device:CPU:0"
            elif device.startswith('gpu'):
                if len(device) == 3:  # just 'gpu'
                    self.device = "/device:GPU:0"
                    # 验证GPU可用性
                    if not gpus:
                        logger.warning("Requested GPU but no GPUs found. Falling back to CPU.")
                        self.device = "/device:CPU:0"
                else:
                    # gpu:N format
                    gpu_idx = device.split(':')[1]
                    self.device = f"/device:GPU:{gpu_idx}"
                    # 验证指定GPU存在
                    if not gpus or int(gpu_idx) >= len(gpus):
                        logger.warning(f"Requested GPU:{gpu_idx} not available. Falling back to CPU.")
                        self.device = "/device:CPU:0"
            else:
                logger.warning(f"Unknown device '{device}', falling back to automatic selection")
                self.device = ""
                self._setup_device()
                return
                
        # 测试选择的设备是否可用
        self._test_device()
        logger.info(f"Using device: {self.device}")
    
    def _test_device(self):
        """测试设备是否正常工作"""
        try:
            with tf.device(self.device):
                # 创建一个小型测试张量
                test_tensor = tf.random.normal([1000, 1000])
                # 执行一些计算
                result = tf.matmul(test_tensor, test_tensor)
                # 强制执行计算以确保设备被使用
                _ = result.numpy()
                logger.info(f"Successfully tested device: {self.device}")
                
                # 如果是GPU设备，尝试打印更多诊断信息
                if "GPU" in self.device:
                    logger.info("GPU memory usage information:")
                    try:
                        # 检查 nvidia-smi 是否可用
                        import subprocess
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv'], 
                            capture_output=True, 
                            text=True
                        )
                        if result.returncode == 0:
                            logger.info(f"NVIDIA-SMI output:\n{result.stdout}")
                    except Exception as e:
                        logger.warning(f"Could not obtain GPU diagnostics: {e}")
        except Exception as e:
            # 如果测试失败并且是GPU设备，尝试退回到CPU
            if "GPU" in self.device:
                logger.warning(f"GPU test failed: {e}. Falling back to CPU.")
                self.device = "/device:CPU:0"
            else:
                # 如果CPU也失败，则抛出异常
                raise RuntimeError(f"Device test failed: {e}")
    
    def _setup_image_size(self):
        """设置图片尺寸"""
        if self.config.img_size == 0:
            self.img_width = 450
            self.img_height = 300
        elif self.config.img_size == 1:
            # 读取图片原始尺寸
            img = cv2.imread(self.config.content_img_path)
            if img is None:
                raise ValueError(f"无法读取内容图片: {self.config.content_img_path}")
            self.img_width = img.shape[1]
            self.img_height = img.shape[0]
        else:
            self.img_width = self.config.img_width
            self.img_height = self.config.img_height
            
        logger.info(f"Image dimensions: {self.img_width} x {self.img_height}")
    
    def _create_output_dir(self):
        """创建输出目录"""
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    def _normalization(self, x):
        """对输入图片进行归一化处理
        
        Args:
            x: 输入图像张量
            
        Returns:
            归一化后的图像张量
        """
        return (x - self.IMAGE_MEAN) / self.IMAGE_STD
    
    def _load_image(self, image_path):
        """加载并预处理图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            预处理后的图片张量
        """
        with tf.device(self.device):
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片不存在: {image_path}")
                
            # 加载文件
            try:
                x = tf.io.read_file(image_path)
                # 解码图片
                x = tf.image.decode_image(x, channels=3, expand_animations=False)
                # 转换为浮点数
                x = tf.cast(x, tf.float32)
                # 修改图片大小
                x = tf.image.resize(x, [self.img_height, self.img_width], 
                                   method=tf.image.ResizeMethod.LANCZOS3)
                x = x / 255.0
                # 归一化处理
                x = self._normalization(x)
                x = tf.expand_dims(x, axis=0)
                return x
            except Exception as e:
                raise RuntimeError(f"加载图片失败 {image_path}: {str(e)}")
    
    def _save_image(self, image, filename):
        """保存生成的图片
        
        Args:
            image: 图像张量
            filename: 保存路径
        """
        with tf.device('/CPU:0'):  # 在CPU上进行图像编码和保存
            x = tf.squeeze(image, 0)
            x = x * self.IMAGE_STD + self.IMAGE_MEAN
            x = x * 255.0
            x = tf.clip_by_value(x, 0, 255)
            x = tf.cast(x, tf.uint8)
            
            if self.config.output_format.lower() == 'png':
                x = tf.image.encode_png(x)
            else:
                x = tf.image.encode_jpeg(x, quality=95)
                
            tf.io.write_file(filename, x)
            logger.info(f"Saved image to {filename}")
    
    def _create_vgg19_model(self, layers):
        """创建并初始化VGG19模型
        
        Args:
            layers: 需要提取的层名列表
            
        Returns:
            keras模型实例
        """
        with tf.device(self.device):
            # 加载imagenet上预训练的vgg19模型
            vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
            # 提取需要被用到的vgg的层的output
            outputs = [vgg.get_layer(layer).output for layer in layers]
            # 使用outputs创建新的模型
            model = tf.keras.Model([vgg.input], outputs)
            # 锁死参数，不进行训练
            model.trainable = False
            return model
    
    def _create_model(self):
        """创建神经风格迁移模型
        
        Returns:
            NeuralStyleTransferModel实例
        """
        return NeuralStyleTransferModel(
            content_layers=self.content_layers,
            style_layers=self.style_layers,
            device=self.device
        )
    
    def _compute_target_features(self):
        """计算目标内容和风格特征"""
        with tf.device(self.device):
            # 确保输入格式正确
            logger.info(f"Computing target features on {self.device}")
            logger.info(f"Content image shape: {self.content_image.shape}")
            logger.info(f"Style image shape: {self.style_image.shape}")
            
            # 计算出目标内容图片的内容特征
            self.target_content_features = self.model(self.content_image)["content"]
            # 计算目标风格图片的风格特征
            self.target_style_features = self.model(self.style_image)["style"]
            
            # 验证特征计算成功
            logger.info(f"Successfully computed content features: {len(self.target_content_features)} layers")
            logger.info(f"Successfully computed style features: {len(self.target_style_features)} layers")
    
    def _init_generated_image(self):
        """初始化生成图像"""
        with tf.device(self.device):
            if self.config.random_init:
                # 随机初始化
                self.generated_image = tf.Variable(
                    tf.random.uniform(
                        (1, self.img_height, self.img_width, 3), 0, 1
                    )
                )
            else:
                # 基于内容图片初始化，添加少量噪声
                self.generated_image = tf.Variable(
                    self.content_image + tf.random.uniform(
                        (1, self.img_height, self.img_width, 3), -0.1, 0.1
                    )
                )
    
    @tf.function
    def _train_step(self):
        """单步训练函数
        
        Returns:
            当前步骤的损失值
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # 计算生成图像的特征
                generated_outputs = self.model(self.generated_image)
                # 计算总损失
                loss = self._compute_total_loss(generated_outputs)
            
            # 计算梯度
            gradients = tape.gradient(loss, self.generated_image)
            # 应用梯度
            self.optimizer.apply_gradients([(gradients, self.generated_image)])
            # 确保像素值在合理范围内 (0-1 after normalization)
            clipped = tf.clip_by_value(self.generated_image, -2, 2)
            self.generated_image.assign(clipped)
            
            return loss
    
    def _compute_content_loss(self, generated_content_features):
        """计算内容损失
        
        Args:
            generated_content_features: 生成图像的内容特征
            
        Returns:
            内容损失值
        """
        content_losses = []
        M = self.img_width * self.img_height
        N = 3
        
        for (gen_feature, factor), (target_feature, _) in zip(
            generated_content_features, self.target_content_features
        ):
            # 计算均方误差
            layer_loss = tf.reduce_sum(tf.square(gen_feature - target_feature))
            # 应用系数
            layer_loss = layer_loss / (2.0 * M * N)
            content_losses.append(layer_loss * factor)
            
        return tf.add_n(content_losses)
    
    def _compute_gram_matrix(self, feature):
        """计算格拉姆矩阵
        
        Args:
            feature: 特征张量
            
        Returns:
            格拉姆矩阵
        """
        # 转置特征，将通道维度放在前面
        transposed = tf.transpose(feature, perm=[2, 0, 1])
        # 将特征重塑为2D矩阵
        flattened = tf.reshape(transposed, (transposed.shape[0], -1))
        # 计算格拉姆矩阵
        gram = tf.matmul(flattened, tf.transpose(flattened))
        return gram
    
    def _compute_style_loss(self, generated_style_features):
        """计算风格损失
        
        Args:
            generated_style_features: 生成图像的风格特征
            
        Returns:
            风格损失值
        """
        style_losses = []
        M = self.img_width * self.img_height
        N = 3
        
        for (gen_feature, factor), (target_feature, _) in zip(
            generated_style_features, self.target_style_features
        ):
            # 计算生成图像的格拉姆矩阵
            gen_gram = self._compute_gram_matrix(gen_feature)
            # 计算目标风格图像的格拉姆矩阵
            target_gram = self._compute_gram_matrix(target_feature)
            # 计算两个格拉姆矩阵之间的均方误差
            layer_loss = tf.reduce_sum(tf.square(gen_gram - target_gram))
            # 应用系数
            layer_loss = layer_loss / (4.0 * (M**2) * (N**2))
            style_losses.append(layer_loss * factor)
            
        return tf.add_n(style_losses)
    
    def _compute_total_loss(self, generated_features):
        """计算总损失
        
        Args:
            generated_features: 生成图像的特征
            
        Returns:
            总损失值
        """
        # 计算内容损失
        content_loss = self._compute_content_loss(generated_features["content"])
        # 计算风格损失
        style_loss = self._compute_style_loss(generated_features["style"])
        # 计算加权总损失
        return (
            content_loss * self.config.content_loss_factor + 
            style_loss * self.config.style_loss_factor
        )
    
    def train(self):
        """执行训练过程"""
        # 提取内容图片文件名（不含扩展名）
        content_name = Path(self.config.content_img_path).stem
        
        # 共训练EPOCHS轮
        for epoch in range(self.config.epochs):
            # 使用tqdm显示进度条
            with tqdm(
                total=self.config.step_per_epoch, 
                desc=f"Epoch {epoch + 1}/{self.config.epochs}"
            ) as pbar:
                # 每轮训练STEPS_PER_EPOCH次
                for step in range(self.config.step_per_epoch):
                    loss = self._train_step()
                    pbar.set_postfix({"loss": f"{float(loss):.4f}"})
                    pbar.update(1)
                
                # 按照设定的间隔保存图片
                if (epoch + 1) % self.config.save_every == 0:
                    output_file = os.path.join(
                        self.config.output_path,
                        f"{content_name}_epoch{epoch + 1}.{self.config.output_format}"
                    )
                    self._save_image(self.generated_image, output_file)


class NeuralStyleTransferModel(tf.keras.Model):
    """神经风格迁移模型类"""
    
    def __init__(
        self,
        content_layers: typing.Dict[str, float],
        style_layers: typing.Dict[str, float],
        device: str = "/CPU:0"
    ):
        """初始化模型
        
        Args:
            content_layers: 内容特征层字典 Dict[层名,加权系数]
            style_layers: 风格特征层字典 Dict[层名,加权系数]
            device: 计算设备
        """
        super(NeuralStyleTransferModel, self).__init__()
        
        self.device = device
        
        # 内容特征层字典
        self.content_layers = content_layers
        # 风格特征层字典
        self.style_layers = style_layers
        
        with tf.device(self.device):
            # 提取需要用到的所有vgg层
            layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
            # 创建layer_name到output索引的映射
            self.outputs_index_map = dict(zip(layers, range(len(layers))))
            # 创建并初始化vgg网络
            self.vgg = self._get_vgg19_model(layers)
    
    def _get_vgg19_model(self, layers):
        """创建并初始化vgg19模型
        
        Args:
            layers: 需要提取的层名列表
            
        Returns:
            keras模型实例
        """
        # 加载imagenet上预训练的vgg19模型
        vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
        # 提取需要被用到的vgg的层的output
        outputs = [vgg.get_layer(layer).output for layer in layers]
        # 使用outputs创建新的模型
        model = tf.keras.Model([vgg.input], outputs)
        # 锁死参数，不进行训练
        model.trainable = False
        return model
    
    # 不对call使用tf.function，避免输入shape不匹配问题
    def call(self, inputs, training=None, mask=None):
        """模型前向传播
        
        Args:
            inputs: 输入图像张量
            training: 是否在训练模式（未使用）
            mask: 掩码（未使用）
            
        Returns:
            包含内容特征和风格特征的字典
        """
        with tf.device(self.device):
            # 确保输入格式正确 - 修复形状不匹配警告
            # 检查输入是否需要添加批次维度
            if len(tf.shape(inputs)) == 3:
                inputs = tf.expand_dims(inputs, 0)
            
            outputs = self.vgg(inputs)
            
            # 分离内容特征层和风格特征层的输出
            content_outputs = []
            for layer, factor in self.content_layers.items():
                # 处理批次维度
                feature = outputs[self.outputs_index_map[layer]]
                if len(tf.shape(feature)) == 4 and tf.shape(feature)[0] == 1:
                    feature = feature[0]  # 取第一个样本，去掉批次维度
                content_outputs.append((feature, factor))
                
            style_outputs = []
            for layer, factor in self.style_layers.items():
                # 处理批次维度
                feature = outputs[self.outputs_index_map[layer]]
                if len(tf.shape(feature)) == 4 and tf.shape(feature)[0] == 1:
                    feature = feature[0]  # 取第一个样本，去掉批次维度
                style_outputs.append((feature, factor))
                
            # 以字典的形式返回输出
            return {"content": content_outputs, "style": style_outputs}


def check_tensorflow_gpu():
    """检查TensorFlow是否正确配置了GPU支持"""
    # 检查TensorFlow是否能看到GPU
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("TensorFlow cannot see any GPUs!")
        logger.info("请检查CUDA和cuDNN是否正确安装并与TensorFlow兼容")
        return False
    
    logger.info(f"TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    
    # 检查GPU是否可用于计算
    try:
        with tf.device('/device:GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            # 强制执行计算
            result = c.numpy()
            del result  # 释放内存
        logger.info("Successfully performed GPU computation test!")
        return True
    except Exception as e:
        logger.error(f"GPU computation test failed: {e}")
        return False

def main():
    """主函数"""
    try:
        # 解析命令行参数
        config = parse_opt()
        
        # 打印配置信息
        logger.info("配置参数:")
        for key, value in vars(config).items():
            logger.info(f"  {key}: {value}")
        
        # 检查TensorFlow GPU支持
        if config.device.lower() == 'gpu' or config.device.lower().startswith('gpu:'):
            gpu_available = check_tensorflow_gpu()
            if not gpu_available:
                if input("GPU无法正常使用。是否使用CPU继续运行？(y/n): ").lower() != 'y':
                    logger.info("用户选择了退出。")
                    exit(0)
                config.device = 'cpu'
                logger.info("切换到CPU模式")
        
        # 创建神经风格迁移实例
        nst = NeuralStyleTransfer(config)
        
        # 执行训练
        nst.train()
        
        logger.info("风格迁移完成！")
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()


'''
Original Author: Steven Chen
Feb. 19, 2022
Optimized Version: March 03, 2025
'''