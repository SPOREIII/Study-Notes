# TensorFlow

官方文档地址：https://tensorflow.google.cn/

## 安装

首先检查你的电脑是否有英伟达显卡，更新你的显卡驱动

- GPU版

  在你的终端或者cmd或者powershell中执行：

  ```bash
  conda update conda
  conda install tensorflow-gpu
  ```

  正常情况下执行命令后会同时下载cudnn和cudatoolkit。如果没有，则

  ```bash
  conda install tensorflow-gpu cudnn cudatoolkit
  ```

  测试是否能进行GPU加速
  
  - 方法一
  
    运行
  
    ```python	
    import tensorflow as tf
    print(tf.test.is_gpu_available())
    ```
  
    若输出`True`则安装成功
  
  - 方法二
  
    运行
  
    ```python
    import tensorflow as tf
    print(tf.config.list_physical_devices())
    ```
  
    若输出
  
    ```bash
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
     PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    ```
  
    则安装成功

## 构建模型之前

- 在tensorflow2.3以上版本中需要加上

  ```python
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  ```

  这三行代码一方面可以使tensorflow按需取用显存，而不是默认模式下的全部占完。同时也可以避免如下的未知错误：

  ```bash
  UnknownError: 2 root error(s) found.
    (0) Unknown:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
  	 [[node encoder/static_handle/res_net/conv2d/Conv2D (defined at /media/E/研究生/program/paper2/code/opt_action_TFT/model/tft.py:308) ]]
  	 [[gradient_tape/encoder/static_handle/embedding_2/embedding_lookup/Reshape/_42]]
    (1) Unknown:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
  	 [[node encoder/static_handle/res_net/conv2d/Conv2D (defined at /media/E/研究生/program/paper2/code/opt_action_TFT/model/tft.py:308) ]]
  0 successful operations.
  0 derived errors ignored. [Op:__inference_training_39566]
  ```

## 自定义训练过程

- 训练两个具有级联结构的模型

    ```python
    x = model_1(inputs)
    x = model_2(x)
    ```

	训练过程如下

    ```python
    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2: #创建两个梯度带
          x = model_1(inputs)
          x = model_2(x)
          loss = loss_function(x)
	
        grads_1 = tape_1.gradient(loss, model_1.trainable_variables)
        grads_2 = tape_2.gradient(loss, model_2.trainable_variables)
	
        optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_variables))
        optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_variables))
	```

## 子类化构建模型

- .build()方法

  `build(self, input_shape)`: This method can be used to create weights that depend on the shape(s) of the input(s), using `add_weight()`. `__call__()` will automatically build the layer (if it has not been built yet) by calling `build()`.

  模型结构需要根据输入tensor的形状进行确定时应该重写此方法，在官方例子中构建了如下的全连接网络：

  ```python
  class Linear(keras.layers.Layer):
      def __init__(self, units=32):
          super(Linear, self).__init__()
          self.units = units
  
      def build(self, input_shape):
          self.w = self.add_weight(
              shape=(input_shape[-1], self.units),
              initializer="random_normal",
              trainable=True,
          )
          self.b = self.add_weight(
              shape=(self.units,), initializer="random_normal", trainable=True
          )
  
      def call(self, inputs):
          return tf.matmul(inputs, self.w) + self.b
  ```

  例子并未说明自定义层具有多输入的情况，做以下实验验证具有多输入的情形：

  定义具有两输入， 两输出， 两层全连接层的自定义层`Multi_dense`

  ```python
  import tensorflow as tf
  import numpy as np
  class Multi_dense(tf.keras.layers.Layer):
      def __init__(self):
          super(Multi_dense, self).__init__()
          self.dense_1 = tf.keras.layers.Dense(3)
          self.dense_2 = tf.keras.layers.Dense(3)
      
      def build(self, inputs_shape):
          print(inputs_shape[0])
          print(inputs_shape[1])
          self.out_dense_1 = tf.keras.layers.Dense(inputs_shape[0][-1])
          self.out_dense_2 = tf.keras.layers.Dense(inputs_shape[1][-1])
      
      def call(self, x):
          x_1 = x[0]
          x_2 = x[1]
          x_1 = self.dense_1(x_1)
          x_2 = self.dense_2(x_2)
          
          x_1 = self.out_dense_1(x_1)
          x_2 = self.out_dense_2(x_2)
          return x_1, x_2
  ```

  构建模型和输入数据

  ```python
  customize_dense = Multi_dense()
  a = np.array([[1,2,4,5]])
  b = np.array([[1,2,3]])
  out_1, out_2 = customize_dense([a,b])
  print(out_1.numpy())
  print(out_2.numpy())
  ```

  打印结果为：

  ```bash
  (1, 4)
  (1, 3)
  [[-0.9102302  1.6765692  6.460695   7.1120706]]
  [[ 0.13991946  0.41044232 -0.27278608]]
  ```

  说明自定义层创建成功，能输出和输入张量相同shape的输出张量。

  `.build()`只能传入一个输入，若有多个输入张量，则需要将它们放进一个`list`中。例如`customize_dense([a,b])`。

## Embedding和Dense的区别

[stack overflow链接](https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer#:~:text=A%20Dense%20layer%20will%20treat%20these%20like%20actual,above%20and%20this%20sentence%3A%20%5B0%2C%202%2C%201%2C%202%5D)、

[TensorFlow API](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Embedding):Embedding turns positive integers (indexes) into dense vectors of fixed size.

对于一个离散输入向量`[0, 2, 1, 2]`

- 全连接网络需要将输入向量转换为one-hot编码：

    ```bash
    one_hot = [[1, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 0, 1]]
    ```

    全连接网络具有一个参数矩阵`w`

    ```bash
    w = [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 0.0, 0.1, 0.2]]
    ```

    忽略激活函数和偏置项，网络输出为`one_hot` $\times$ `w`

    ```bash
    [[1 * 0.1 + 0 * 0.5 + 0 * 0.9, 1 * 0.2 + 0 * 0.6 + 0 * 0.0, 1 * 0.3 + 0 * 0.7 + 0 * 0.1, 1 * 0.4 + 0 * 0.8 + 0 * 0.2],
     [0 * 0.1 + 0 * 0.5 + 1 * 0.9, 0 * 0.2 + 0 * 0.6 + 1 * 0.0, 0 * 0.3 + 0 * 0.7 + 1 * 0.1, 0 * 0.4 + 0 * 0.8 + 1 * 0.2],
     [0 * 0.1 + 1 * 0.5 + 0 * 0.9, 0 * 0.2 + 1 * 0.6 + 0 * 0.0, 0 * 0.3 + 1 * 0.7 + 0 * 0.1, 0 * 0.4 + 1 * 0.8 + 0 * 0.2],
     [0 * 0.1 + 0 * 0.5 + 1 * 0.9, 0 * 0.2 + 0 * 0.6 + 1 * 0.0, 0 * 0.3 + 0 * 0.7 + 1 * 0.1, 0 * 0.4 + 0 * 0.8 + 1 * 0.2]]
    ```

    ​	=

    ```
    [[0.1, 0.2, 0.3, 0.4],
     [0.9, 0.0, 0.1, 0.2],
     [0.5, 0.6, 0.7, 0.8],
     [0.9, 0.0, 0.1, 0.2]]
    ```

- Embedding直接以整数向量为输入

  假设Embedding具有同样的参数矩阵`w`:

  ```bash
  w = [[0.1, 0.2, 0.3, 0.4],
       [0.5, 0.6, 0.7, 0.8],
       [0.9, 0.0, 0.1, 0.2]]
  ```

  层输出为：

  ```bash
  [w[0],
   w[2],
   w[1],
   w[2]]
  ```

  =

  ```bash
  [[0.1, 0.2, 0.3, 0.4],
   [0.9, 0.0, 0.1, 0.2],
   [0.5, 0.6, 0.7, 0.8],
   [0.9, 0.0, 0.1, 0.2]]
  ```

输出结果相同

## Debug

- 训练loss出现nan

  请先检查你的数据集是否存在空缺

  ```python
  np.any(np.isnan(dataset))
  ```

- 自定义模型训练过程，每次验证损失不一样

  自定义了模型的训练过程，发现模型训练完成后在验证集上的`loss`多次重复运行每次输出都不一样。

  原因是构建验证数据集用了如下代码：

  ```python
  val_data = tf.data.Dataset.from_tensor_slices({'x':(val_dataset['some_data_1'],
                                                      val_dataset['some_data_1']),
                                                 'y':y_val_data})
  val_data = val_data.shuffle(buffer_size=1024).batch(64)
  ```

  文档中关于`tf.data.Dataset.shuffle()`的描述为
  
  Randomly shuffles the elements of this dataset.
  
  This dataset fills a buffer with `buffer_size` elements, then randomly samples elements from this buffer, replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
  
  For instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1,000, then `shuffle` will initially select a random element from only the first 1,000 elements in the buffer. Once an element is selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element, maintaining the 1,000 element buffer.
  
  按道理来讲`.shuffle()`只会改变传入数据的顺序而不会对数据重复采样或者少采样。但在实作中发现将`.shuffle()`删除后解决问题。
  
  ```python
  val_data = tf.data.Dataset.from_tensor_slices({'x':(val_dataset['some_data_1'],
                                                      val_dataset['some_data_1']),
                                                 'y':y_val_data})
  val_data = val_data.batch(64)
  ```
  
  

