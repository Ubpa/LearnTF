# 01_03 回归

在*回归*问题中，我们的目标是预测连续值的输出，如价格或概率。将此与*分类*问题进行对比，我们的目标是预测离散标签（例如，图片包含苹果或橙色）。

这款笔记本采用了经典的[Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)数据集，并建立了一个模型来预测20世纪70年代末和80年代初汽车的燃油效率。为此，我们将为模型提供该时间段内许多模型的描述。此描述包括以下属性：气缸，排量，马力和重量。

Seaborn是一个基于[matplotlib](https://matplotlib.org/)的Python数据可视化库。它提供了一个高级界面，用于绘制有吸引力且信息丰富的统计图形。

```bash
# Use seaborn for pairplot
pip install -q seaborn
```

```python
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1.12.0
print(tf.__version__)
```

## 1. Auto MPG数据集

该数据集可从[UCI Machine Learning Repository](https://archive.ics.uci.edu/) 获得

### 1.1 获取数据

首先下载数据集。

```python
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)
```

使用pandas导入它

```python
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())
```

|      | MPG  | Cylinders | Displacement | Horsepower | Weight | Acceleration | Model Year | Origin |
| ---- | --------- | ------------ | ---------- | ------ | ------------ | ---------- | ------ | ---- |
| 393  | 27.0      | 4            | 140.0      | 86.0   | 2790.0       | 15.6       | 82     | 1    |
| 394  | 44.0      | 4            | 97.0       | 52.0   | 2130.0       | 24.6       | 82     | 2    |
| 395  | 32.0      | 4            | 135.0      | 84.0   | 2295.0       | 11.6       | 82     | 1    |
| 396  | 28.0      | 4            | 120.0      | 79.0   | 2625.0       | 18.6       | 82     | 1    |
| 397  | 31.0      | 4            | 119.0      | 82.0   | 2720.0       | 19.4       | 82     | 1    |

### 1.2 清理数据

数据集包含一些未知值。

```python
print(dataset.isna().sum())
```

为了让这个初始教程简单些，所以直接删除那些行。

```python
dataset = dataset.dropna()
```

该`"Origin"`列实际上是分类的，而不是数字的。所以把它转换为one-hot：

```python
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())
```

### 1.3 将数据拆分为训练集和测试集

```python
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
```

### 1.4 检查数据

绘制散点图，kde是 kernel density estimation，**概率论**中用来估计未知的**密度函数**，属于非参数检验方法

```python
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()
```

![PNG](assets/output_20_1.png)

另请查看整体统计数据：

```python
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
```

### 1.5 从标签中分离特征

将目标值或“标签”与特征分开。此标签是您将训练模型进行预测的值。

```python
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
```

### 1.6 规范化数据

将使用不同尺度和范围的特性规范化是一种很好的做法。虽然该模型在没有特征归一化的情况下可能会收敛，但它使得训练变得更加困难，并且使得得到的模型依赖于输入中使用的单元的选择。

> **Note:** 我们有意只使用来自训练集的统计数据，这些统计数据也将用于评估。这样模型就没有关于测试集的任何信息。

```python
normed_train_data = (train_dataset - train_stats['mean']) / train_stats['std']
normed_test_data = (test_dataset - train_stats['mean']) / train_stats['std']
```

## 2. 模型

### 2.1 建立模型

让我们建立我们的模型。在这里，我们将使用`Sequential`，它具有两个密集连接的隐藏层的模型，以及返回单个连续值的输出层。模型构建步骤包含在一个函数中`build_model`，因为我们稍后将创建第二个模型。

```python
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```

```python
model = build_model()
```

### 2.2 检查模型

```python
model.summary()
```

现在试试这个模型。从训练数据中取`10`个样例的一组，并调用`model.predict`。

```python
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)
```

### 2.3 训练模型

该模型经过1000个epochs的训练，并记录对象的训练和验证准确性`history`。

```python
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
```

使用存储在`history`对象中的统计信息可视化模型的训练进度。

```python
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  hist.tail()
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

plot_history(history)
plt.show()
```

![PNG](assets/output_42_0.png)

![PNG](assets/output_42_1.png)

该图显示几百个epochs后的验证错误几乎没有改善，甚至降低。让我们更新`model.fit`方法，以便在验证分数没有提高时自动停止培训。我们将使用一个*回调*来测试每个时代的训练条件。如果经过一定数量的时期而没有显示出改进，则自动停止训练。

```python
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
```

让我们看看模型在**测试**集上是如何执行的，我们在训练模型时没有使用它：

```python
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
```

### 2.4 作出预测

```python
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100, 100], [-100, 100])
plt.show()
```

![PNG](assets/output_48_0.png)

```python
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()
```

![PNG](assets/output_49_0.png)

## 3. 结论

这款笔记本介绍了一些处理回归问题的技巧。

- 均方误差（MSE）是用于回归问题的常见损失函数（不同于分类问题）。
- 同样，用于回归的评估指标也不同于分类。常见的回归指标是平均绝对误差（MAE）。
- 当输入数据要素具有不同范围的值时，应单独缩放每个要素。
- 如果训练数据不多，则选择隐藏层较少的小型网络，以避免过度拟合。
- 早期停止是防止过度拟合的有用技术。