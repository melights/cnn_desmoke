from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DesmokeNet(object):
    """monodepth model"""

    def __init__(self, smoke_images, original_images):
        self.smoke_images = smoke_images
        self.original_images = original_images
        self.build_net()
        self.build_losses()
        self.build_summaries()
    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def build_net(self):
        #set convenience functions
        conv = self.conv
        upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.smoke_images,  32, 7) # H/2
            conv2 = self.conv_block(conv1,             64, 5) # H/4
            conv3 = self.conv_block(conv2,            128, 3) # H/8
            conv4 = self.conv_block(conv3,            256, 3) # H/16
            conv5 = self.conv_block(conv4,            512, 3) # H/32
            conv6 = self.conv_block(conv5,            512, 3) # H/64
            conv7 = self.conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.out4 = conv(iconv4,  3, 3, 1)
            uout4  = self.upsample_nn(self.out4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, uout4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.out3 = conv(iconv3,  3, 3, 1)
            uout3  = self.upsample_nn(self.out3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, uout3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.out2 = conv(iconv2,  3, 3, 1)
            uout2  = self.upsample_nn(self.out2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, uout2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.out1 = conv(iconv1,  3, 3, 1)
    

    def build_losses(self):
            self.original_pyramid = self.scale_pyramid(self.original_images, 4)
            self.desmoked  = [self.out1, self.out2, self.out3, self.out4]

            self.l1 = [tf.abs( self.desmoked[i] - self.original_pyramid[i]) for i in range(4)]
            self.l1_loss  = tf.add_n([tf.reduce_mean(l) for l in self.l1])
 
            #self.ssim_left = [self.SSIM( self.desmoked[i],  self.original_pyramid[i]) for i in range(4)]
            #self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]

            #self.image_loss_left  = [0.85 * self.ssim_loss_left[i]  + (1 - 0.85) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            #self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            self.total_loss = self.l1_loss 

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
                #tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i])
            tf.summary.scalar('l1_loss' , self.l1_loss)
                #tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i])
            tf.summary.image('smoke',  self.smoke_images,   max_outputs=4)
            tf.summary.image('original', self.original_images,  max_outputs=4)
            tf.summary.image('desmoked', self.desmoked[0],  max_outputs=4)

