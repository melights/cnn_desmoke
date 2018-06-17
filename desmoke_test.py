import sys
import tensorflow as tf
import scipy
from desmoke_net import *

#Usage: python desmoke_test.py input_img model_path out_img
input_image = scipy.misc.imread(sys.argv[1])
input_image = input_image[:,:,:3]
input_image = scipy.misc.imresize(input_image, [128, 256], interp='lanczos')
input_image = input_image.astype(np.float32) / 255
input_image=tf.expand_dims(input_image,0)
model = DesmokeNet(input_image,input_image)
sess = tf.Session()
train_saver = tf.train.Saver()
train_saver.restore(sess, sys.argv[2])
desomked_img = sess.run(model.out1)
scipy.misc.imsave(sys.argv[3],desomked_img[0])
print('done.')
