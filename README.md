# TF-Learn
该项目是我自己用tensorflow、keras实现的一些分类（目标检测尚在写）的例子，用来学习tensorflow等框架的使用。

目录说明

一、Basenet

   __init__.py 为了方便其他文件调用，一个可以为空的文件
   
   Basenet.py  使用tensorflow的基础api: tf.nn下的各种函数写的伸进网络基础层的实现文档，下面用于分类的网络VGG、Resnet都有用到这个文件中的函数来实现。
   
二、CRNN-TF

   这是图像中文字识别的CRNN实现，可以在其他的分类任务十分熟练之后再看这个。这个文档下的文档说明也将再后面说明。
   
三、MNIST-TF

   mnist.py  用tensorflow实现的卷积网络并在MNIST数据集上训练文件。
   MNIST_data  MNIST数据集。

四、Resnets_models
   Resnets_TF.py
五、vgg16_models       用
VGG16_TF.py
vgg16_keras.py
vgg16_tflayers.py

format_conversion.py
restore.py
test.py
test_lr.py
train.py
train_vgg_keras.py
utils.py

