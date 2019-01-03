# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization,concatenate
import tensorflow as tf
import keras.backend as K
import cfg

num_cores = 1
GPU = 0
CPU = 1
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)


class PixelLink:

    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(720,1280, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      # weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]
        self.f.insert(0,vgg16.get_layer('block5_conv3').output)
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num

    def upsample(self,x):
        return UpSampling2D((2, 2))(x)

    def build_network(self):
        block1 = self.f[-1]     # 1/2
        block2 = self.f[-2]     # 1/4
        block3 = self.f[-3]     # 1/8
        block4 = self.f[-4]     # 1/16
        block5 = self.f[-5]     # 1/16

        fc1 = Conv2D(512, (3, 3),activation='relu',padding='same',name='fc1')(block5)
        fc2 = Conv2D(512, (3, 3),activation='relu',padding='same',name='fc2')(fc1)

        link1 = Conv2D(16, (1, 1),activation='relu',padding='same',name='link1')(fc2)

        link2 = Conv2D(16, (1, 1),activation='relu',padding='same',name='link2')(block5)
        link2 = concatenate([link1,link2],axis=-1)          # 1/16
        # link2 = self.upsample(link2)        # 1/8

        link3 = Conv2D(16, (1, 1),activation='relu',padding='same',name='link3')(block4)
        link3 = concatenate([link2, link3], axis=-1)
        link3 = self.upsample(link3)  # 1/8

        link4 = Conv2D(16, (1, 1),activation='relu',padding='same',name='link4')(block3)
        link4 = concatenate([link3, link4], axis=-1)
        link4 = self.upsample(link4)  # 1/4

        link5 = Conv2D(16, (1, 1), activation='relu', padding='same', name='link5')(block2)
        link5 = concatenate([link4, link5], axis=-1)            # 1/4

        link_pre = Conv2D(16, (1, 1), activation='softmax', padding='same', name='link_pre')(link5)

        #-------------------------------------------------------------#

        cls1 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls1')(fc2)

        cls2 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls2')(block5)
        cls2 = concatenate([cls1, cls2], axis=-1)
        # cls2 = self.upsample(cls2)  # 1/8

        cls3 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls3')(block4)
        cls3 = concatenate([cls2, cls3], axis=-1)
        cls3 = self.upsample(cls3)  # 1/4

        cls4 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls4')(block3)
        cls4 = concatenate([cls3, cls4], axis=-1)
        cls4 = self.upsample(cls4)  # 1/2

        cls5 = Conv2D(2, (1, 1), activation='relu', padding='same', name='cls5')(block2)
        cls5 = concatenate([cls4, cls5], axis=-1)

        cls_pre = Conv2D(2, (1, 1), activation='softmax', padding='same', name='cls_pre')(cls5)

        east_detect = Concatenate(axis=-1,name='east_detect')([cls_pre,link_pre])
        return Model(inputs=self.input_img, outputs=east_detect)


if __name__ == '__main__':
    pl = PixelLink()
    pl_network = pl.build_network()
    pl_network.summary()
