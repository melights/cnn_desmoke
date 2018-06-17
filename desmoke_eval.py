import sys,os
import tensorflow as tf
import scipy
from desmoke_net import *
import timeit
import numpy as np
original_image_path='/home/long/data/daVinci/train/image_0/'
smoke_image_path='/home/long/data/render_smokes2/image/3/'
filenames_file='test.txt'

img  = tf.placeholder(tf.float32, [1, 128, 256, 3])
model = DesmokeNet(img,img)
sess = tf.Session()
train_saver = tf.train.Saver()
train_saver.restore(sess, sys.argv[1])

error=[]
f = open(filenames_file,'r')
for line in f:
    smoke_image = scipy.misc.imread(smoke_image_path+line.rstrip())
    smoke_image = smoke_image[:,:,:3]
    smoke_image = scipy.misc.imresize(smoke_image, [128, 256], interp='lanczos')
    smoke_image = smoke_image.astype(np.float32) / 255
    smoke_image=np.expand_dims(smoke_image,0)
    start = timeit.default_timer()
    desomked_img = sess.run(model.out1, feed_dict={img: smoke_image})
    stop = timeit.default_timer()
    ori_image = scipy.misc.imread(original_image_path+line.rstrip())
    ori_image = ori_image[:,:,:3]
    ori_image = scipy.misc.imresize(ori_image, [128, 256], interp='lanczos')
    ori_image = ori_image.astype(np.float32) / 255
    ori_image=np.expand_dims(ori_image,0)
    loss=np.mean(np.abs(desomked_img-ori_image))
    error.append(loss)
    print("File:{} | Time:{:.3f}s | Loss:{:.3f} |  | ".format(line.rstrip(),stop - start,loss))
print("Evaluation result:{:.5f} |  | ".format(np.mean(error)))

