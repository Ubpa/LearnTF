# 04. 变分自解码器

## 1. 原理

> 参考链接
>
> [1] [变分自编码器VAE：原来是这么一回事 | 附开源代码](https://zhuanlan.zhihu.com/p/34998569) 
>
> [2] [【啄米日常】7：Keras示例程序解析（4）：变分编码器VAE](https://zhuanlan.zhihu.com/p/25269592) 

我们想要用 $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},I)​$ 来生成 $\boldsymbol{x}​$，$p(\boldsymbol{x})=f(\boldsymbol{z})​$。这里 $\boldsymbol{z}​$ 叫**隐变量**。这个 $f​$ 就是 **生成器** 需要通过学习来获得。

怎么学这个 **生成器** 呢？我们通过 **自解码器** 的方式来学习，即用 $\boldsymbol{x}​$ 生成 $\boldsymbol{z}​$，再用$\boldsymbol{z}​$ 重构 $\boldsymbol{x}​$。

一个 $\boldsymbol{x}$，会对应一个 $\boldsymbol{z}$ 的分布，假设其符合高斯分布，则有$p(\boldsymbol{z}|\boldsymbol{x})=\mathcal{N}(\boldsymbol{z}|\mu,\text{diag}(\boldsymbol{\sigma}^2))$。但是我们并不知道$\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma^2}$。所以得靠网络来学习$\boldsymbol{\mu}(\boldsymbol{x})$，$\boldsymbol{\sigma}^2(\boldsymbol{x})$。*注意，这里一个 $\boldsymbol{x}$ 对应一个 $\boldsymbol{\mu}$和$\boldsymbol{\sigma}^2$，并不是所有的 $\boldsymbol{\boldsymbol{x}}$ 对应同一个 $\boldsymbol{\mu}$和$\boldsymbol{\sigma}^2$，更不是所有的 $\boldsymbol{\boldsymbol{x}}$ 对应同一个 $\boldsymbol{\mu}=\boldsymbol{0}$和$\boldsymbol{\sigma}^2=\boldsymbol{1}​$*。

作为解码器，自然的，我们要求**重构的$\boldsymbol{x}​$ 和原$\boldsymbol{x}​$ 相近**。因为这里的 $\boldsymbol{z}​$ 是一个随机变量，所以我们改为要求**重构的$\boldsymbol{x}​$ 的期望和原$\boldsymbol{x}​$相近**。那么我们就要最小化
$$
\|f(\mathbb{E}_{z\in\mathcal{N}(\boldsymbol{\mu}(\boldsymbol{x}),\text{diag}(\boldsymbol{\sigma}^2(\boldsymbol{x}))})-\boldsymbol{x}\|_2^2
$$
这个期望是一个积分，可以采用蒙特卡罗积分法，采样可以用**重参数化**的方式进行，即
$$
\boldsymbol{z}=\boldsymbol{\mu}(\boldsymbol{x})+\boldsymbol{\epsilon}\odot\boldsymbol{\sigma}(\boldsymbol{x}),\epsilon\sim\mathcal{N}(\boldsymbol{0},I)
$$
但我们还希望用一个 $\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},I)​$ 去生成 $\boldsymbol{x}​$，因此要求函数 $\boldsymbol{\mu}(\boldsymbol{x})​$ 和$\boldsymbol{\sigma}^2(\boldsymbol{x})​$ 尽量往 $\boldsymbol{\mu}=\boldsymbol{0}​$ 和 $\boldsymbol{\sigma}^2=\boldsymbol{1}​$ 靠，否则我们用这样的 $\boldsymbol{z}​$ 很难生成偏离这个分布很大的数据。这个可以通过最小化 KL 散度来达到。
$$
KL(\mathcal{N}(\boldsymbol{\mu},\text{diag}(\boldsymbol{\sigma}^2)\|\mathcal{N}(\boldsymbol{0},I))=\sum_{i=1}^n\frac{1}{2}(-\log \sigma_i^2+\mu_i^2+\sigma_i^2-1)
$$

主要结构如下

![img](assets/v2-36c7da0b2fe37bd021699532a2cff1e8_hd.jpg)

## 2. 实现

```python
'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models, data, batch_size=128, model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = x_train.shape[1:]
original_dim = input_shape[0]*input_shape[1]
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 5

# VAE model = encoder + decoder
# build encoder model

inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Reshape(target_shape=(7*7*32,))(x)
x = Dense(units=intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(7*7*32, activation='relu')(x)
x = Reshape(target_shape=(7,7,32,))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

models = (encoder, decoder)
data = (x_test, y_test)

# VAE loss = xent_loss + kl_loss
reconstruction_loss = binary_crossentropy(inputs, outputs)
#reconstruction_loss *= original_dim
reconstruction_loss = K.sum(reconstruction_loss, axis=1)
reconstruction_loss = K.sum(reconstruction_loss, axis=1)

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

w = [10, 1]

vae_loss = K.mean(w[0] * reconstruction_loss + w[1] * kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae,
           to_file='vae_mlp.png',
           show_shapes=True)


# train the autoencoder
vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))

plot_results(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")
```

![04_z](assets/04_z.png)

$\boldsymbol{z}​$ 把各种数字一定程度上分开了，分的不够开可能是维度2太少了，用生成器生成数字，如下

![04_x](assets/04_x.png)

