# -*- coding: utf-8 -*-

# ------------------------------------
# 
# 01 词云
# 
# ------------------------------------

from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import jieba
import jieba.analyse
import random
from PIL import Image

data_root = '..\\..\\datasets\\01\\'

# ------------------------------------
# 1. 一个简单的例子
# ------------------------------------

# 打开文本
text0 = open(data_root + 'constitution.txt').read()
# 生成对象
wc = WordCloud().generate(text0)

# 显示词云
def plot_wc(wc):
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

plot_wc(wc)

# 保存到文件
wc.to_file('wordcloud_01.png')

# ------------------------------------
# 2. 中文词云
# ------------------------------------


text1 = open(data_root + 'xyj.txt', encoding='utf8').read()

# -- 2.1 不分词 --

wc = WordCloud(font_path = data_root + 'Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text1)

plot_wc(wc)

wc.to_file('wordcloud_02_01.png')

# -- 2.2 分词 --

text2 = ' '.join(jieba.cut(text1))

wc = WordCloud(font_path = data_root + 'Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text2)

plot_wc(wc)

wc.to_file('wordcloud_02_02.png')

# ------------------------------------
# 3. 蒙版
# ------------------------------------

# -- 3.1 简单例子 -- 

mask = np.array(Image.open(data_root+"black_mask.png"))

wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text2)

plot_wc(wc)

wc.to_file('wordcloud_03_01.png')

# -- 3.2 蒙版颜色 --

mask = np.array(Image.open(data_root+"color_mask.png"))

wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

plot_wc(wc)

wc.to_file('wordcloud_03_02.png')

# -- 3.3 纯色 --

# 颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
	s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
	print(s)
	return s

mask = np.array(Image.open(data_root+"black_mask.png"))
wc = WordCloud(color_func=random_color, mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

plot_wc(wc)

wc.to_file('wordcloud_03_03.png')

# ------------------------------------
# 4. 蒙版
# ------------------------------------

freq = jieba.analyse.extract_tags(text1, topK=200, withWeight=True)
print(freq[:20])
freq = {i[0]: i[1] for i in freq}

mask = np.array(Image.open(data_root+"color_mask.png"))
wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate_from_frequencies(freq)

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

plot_wc(wc)

wc.to_file('wordcloud_04.png')
