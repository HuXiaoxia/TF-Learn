# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:19:39 2019

@author: Administrator
"""
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input

from vgg16_models.vgg16_keras import vgg
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
# parameters
BATCH_SIZE = 64
# Epoch 数
NUM_EPOCHS = 20
# 模型输出路径
WEIGHTS_FINAL = 'model-vgg16-final.h5'

path_flower='./Dataset/flowers/'
path_17flower='./Dataset/17flowers/'
path_bk ='./Dataset/Datasets_horror_normal/'

path_train = path_17flower

num_class  =len(os.listdir(path_train+'train'))



x = Input(name='input_img',shape=(224, 224, 3),dtype='float32')
vgg16 = vgg(x,num_class,weights='imagenet',locked_layers=False).net_out
vgg16.summary()



train_datagen = ImageDataGenerator(rescale=1./255)
 
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
 
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        path_train+'train',  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        shuffle = True,
        batch_size=BATCH_SIZE,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
 
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path_train+'val',
        target_size=(224, 224),
        class_mode='categorical',
        shuffle=True,
        batch_size=BATCH_SIZE)
if path_train==path_bk:
    # 输出各类别的索引值
    if os.path.exists('./labels_index_bk.txt'):
        os.remove('./labels_index_bk.txt')
    for cls, idx in train_generator.class_indices.items():
        print('{}:{}'.format(cls,idx ))
        with open('labels_index_bk.txt','a') as f:
            f.write('{}:{}'.format(cls,idx ))
            f.write('\n')
elif path_train==path_flower:
    # 输出各类别的索引值
    if os.path.exists('./labels_index_flowers.txt'):
        os.remove('./labels_index_flowers.txt')
    for cls, idx in train_generator.class_indices.items():
        print('{}:{}'.format(cls,idx ))
        with open('labels_index_flowers.txt','a') as f:
            f.write('{}:{}'.format(cls,idx ))
            f.write('\n')
elif path_train == path_17flower:
    # 输出各类别的索引值
    if os.path.exists('./labels_index_17flowers.txt'):
        os.remove('./labels_index_17flowers.txt')
    for cls, idx in train_generator.class_indices.items():
        print('{}:{}'.format(cls,idx ))
        with open('labels_index_17flowers.txt','a') as f:
            f.write('{}:{}'.format(cls,idx ))
            f.write('\n')
    
vgg16.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
vgg16.fit_generator(train_generator,
                        steps_per_epoch = train_generator.samples // BATCH_SIZE,
                        validation_data = validation_generator,
                        validation_steps = validation_generator.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# 存储训练好的模型
vgg16.save(WEIGHTS_FINAL)
