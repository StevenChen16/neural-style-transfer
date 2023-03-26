# -*- coding: utf-8 -*-

# 内容特征层及loss加权系数
CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
# 风格特征层及loss加权系数
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}
