# 01 词云

## 0. 简介

词云是一种数据呈现方式

使用开源的 [wordcloud](https://github.com/amueller/word_cloud) 来生成图云，文档为 [wordcloud_references](http://amueller.github.io/word_cloud/references.html) 

中文使用 `jieba` 来分词

安装依赖项

```bash
pip install numpy wordcloud matplotlib jieba Pillow
```

导入依赖项，设置路径

```python
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
import jieba
import jieba.analyse
import random
from PIL import Image

data_root = '..\\..\\datasets\\01\\'
```

## 1. 一个简单的例子

使用 `constitution.txt` 来生成图云

由于英文单词之间有空格分隔，因此大多不需要额外的处理

```python
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
wc.to_file( 'wordcloud_01.png')
```

![03_wordcloud_01](assets/03_wordcloud_01.png)

## 2. 中文词云

使用 `xyj.txt` （西游记）来分词

首先初始化

```python
import jieba

# 打开文本
text1 = open(data_root + 'xyj.txt', encoding='utf8').read()
```



### 2.1 不分词

中文一般需要经过分词处理，先看不分词的效果

```python
wc = WordCloud(font_path = data_root + 'Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text1)

plot_wc(wc)

wc.to_file( 'wordcloud_02_01.png')
```

![03_wordcloud_02](assets/03_wordcloud_02_01.png)

可以看到结果中会出现各种双字、三字和四字等，但很多并不是合理的词语

### 2.2 分词

使用 `jieba` 来分词

```python
text2 = ' '.join(jieba.cut(text1))

wc = WordCloud(font_path = data_root + 'Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text2)

plot_wc(wc)

wc.to_file( 'wordcloud_02_02.png')
```

![03_wordcloud_02_02](assets/03_wordcloud_02_02.png)

可以看到生成的词云里，基本上都是合理的词语了

## 3. 蒙版

### 3.1 简单例子

这⾥将 mask 翻译为蒙版，是因为感觉它和 Photoshop 中蒙版的作⽤很类似。 

使⽤蒙版之后，可以根据提供的蒙版图⽚，⽣成指定形状的词云

`mask` 如下

![03_wordcloud_black_mask](assets/03_wordcloud_black_mask.png)

```python
mask = np.array(Image.open(data_root+"black_mask.png"))

wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text2)

plot_wc(wc)

wc.to_file( 'wordcloud_03_01.png')
```

![03_wordcloud_03](assets/03_wordcloud_03.png)

### 3.2 蒙版颜色

词云的颜⾊可以从蒙版中抽取，使⽤ `ImageColorGenerator()` 即可。 

![03_wordcloud_color_mask](assets/03_wordcloud_color_mask.png)

```python
mask = np.array(Image.open(data_root+"color_mask.png"))

wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

plot_wc(wc)

wc.to_file( 'wordcloud_03_02.png')
```

![wordcloud_03_02](assets/wordcloud_03_02.png)

### 3.3 纯色

当然也可以设置为纯色，增加一个配色函数即可

```python
# 颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
	s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
	print(s)
	return s

mask = np.array(Image.open(data_root+"black_mask.png"))
wc = WordCloud(color_func=random_color, mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

plot_wc(wc)

wc.to_file( 'wordcloud_03_03.png')
```

![wordcloud_03_03](assets/wordcloud_03_03.png)

## 4. 精细控制

如果希望精细地控制词云中出现的词，以及每个词的⼤⼩，可以尝试 `generate_from_frequencies()`，包括两个参数

- frequencies：⼀个字典，⽤于指定词和对应的⼤⼩；

- max_font_size：最⼤字号，默认为 `None`。

`generate() = process_text() + generate_from_frequencies()` 

```python
freq = jieba.analyse.extract_tags(text1, topK=200, withWeight=True)
print(freq[:20])
freq = {i[0]: i[1] for i in freq}

mask = np.array(Image.open(data_root+"color_mask.png"))
wc = WordCloud(mask=mask, font_path=data_root+'Hiragino.ttf', mode='RGBA', background_color=None).generate_from_frequencies(freq)

image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

plot_wc(wc)

wc.to_file( 'wordcloud_04.png')
```

![wordcloud_04](assets/wordcloud_04.png)

