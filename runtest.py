import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.mlcompute import  mlcompute
mlcompute.set_mlc_device('gpu')
#验证集和测试集结果生成
#加载图片
def load_img(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32)
    image=image/31
    return image
#本地测试集测试
imgs_path_t=glob.glob('val/*jpg')
print(imgs_path_t)
len_test=len(imgs_path_t)
print("test lens:"+str(len_test))
#加载模型
model_test=tf.keras.models.load_model('/Users/gutao/Desktop/p/cv21b/mds/m1.h5')
a=0
f = open("171250575-a.txt",'w')
for i in imgs_path_t:
    a=a+1
    test_img=i
    test_tensor=load_img(test_img)
    test_tensor=tf.expand_dims(test_tensor,axis=0)
    pred=model_test.predict(test_tensor)
    pred=int(np.argmax(pred))
    res=i+' '+str(pred)
    res=res.split('/')[1]+'\n'
    f.write(res)
    print(a/len_test)
f.close()