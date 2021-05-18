# -*- coding: utf-8 -*-
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import  mlcompute
mlcompute.set_mlc_device(device_name="gpu")
#测试集获取
dic={}
f = open("val_anno.txt")
line = f.readline()
while line:
    str1=line.split(' ')[0]
    str2=int(line.split(' ')[1])
    dic[str1]=str2
    line=f.readline()
f.close()
imgs_path_t=glob.glob('val/*jpg')
labels_t=[dic[img_path_t.split('/')[1]] for img_path_t in imgs_path_t]
#训练集获取
imgs_path=glob.glob('train/*/*jpg')
all_labels_name=[img_path.split('/')[1] for img_path in imgs_path]
labels_name=np.unique(all_labels_name)

all_labels=[int(name) for name in all_labels_name]
#乱序化
np.random.seed(2021)
random_index=np.random.permutation(len(imgs_path))

imgs_path=np.array(imgs_path)[random_index]
all_labels=np.array(all_labels)[random_index]
imgs_path_t=np.array(imgs_path_t)
labels_t=np.array(labels_t)
#训练集测试集划分
train_path=imgs_path
train_labels=all_labels

test_path=imgs_path_t
test_labels=labels_t

print("train size = "+str(len(train_path)))
print("test size = "+str(len(test_path)))
train_ds=tf.data.Dataset.from_tensor_slices((train_path,train_labels))
test_ds=tf.data.Dataset.from_tensor_slices((test_path,test_labels))
#读取图片方法
def load_img(path,label):
    image=tf.io.read_file(path)
    image=tf.image.decode_jpeg(image,channels=3)
    image=tf.image.resize(image,[32,32])
    image=tf.cast(image,tf.float32)
    image=image/32
    return image,label
#多线程读入
train_ds=train_ds.map(load_img)
test_ds=test_ds.map(load_img)
print (train_ds)
BATCH_SIZE=32
train_ds=train_ds.repeat().shuffle(1000).batch(BATCH_SIZE)
test_ds=test_ds.batch(BATCH_SIZE)
print (test_ds)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'
                           ,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(80),
])
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'])
train_count=len(train_path)
test_count=len(test_path)
steps_per_epoch=train_count//BATCH_SIZE
validation_steps=test_count//BATCH_SIZE
history=model.fit(train_ds,epochs=30,steps_per_epoch=steps_per_epoch,
                  validation_data=test_ds,validation_steps=validation_steps)
#模型的保存
model.save('/Users/gutao/Desktop/p/cv21b/mds/m1.h5')
#
#
#
#
