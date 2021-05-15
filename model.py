from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2


channel_dim=2
dropout_probability = 0.2
weight_decay=0.0005

class WideResNet:

  def __init__(self, input_filter, output_filter, stride):
    self.input_filter = input_filter
    self.output_filter = output_filter
    self.stride = stride
  
  def __call__(self, input_val):

    strides_params = [self.stride, (1,1)]

    for i, val in enumerate(strides_params):

      if i == 0:
        if self.output_filter != self.input_filter:
          logit = layers.BatchNormalization(axis=channel_dim)(input_val)
          logit = layers.Activation('relu')(logit)
          conv_out = layers.Conv2D(self.output_filter, (3, 3), strides=val, padding='same',
                                   kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(logit)
          shortcut = layers.Conv2D(self.output_filter, (3, 3), strides=val, padding='same',
                                   kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(logit)
                    
        else:
          logit = layers.BatchNormalization(axis=channel_dim)(input_val)
          logit = layers.Activation('relu')(logit)
          if dropout_probability > 0.0:
            logit = layers.Dropout(rate=dropout_probability)(logit)
          conv_out = layers.Conv2D(self.output_filter, (3,3), strides=val, padding='same',
                                   kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(logit)
          shortcut = input_val
          
      else:
        logit = layers.BatchNormalization(axis=channel_dim)(conv_out)
        logit = layers.Activation('relu')(logit)
        conv_out = layers.Conv2D(self.output_filter, (3,3), strides=val, padding='same',
                                 kernel_initializer='glorot_uniform', kernel_regularizer=l2(weight_decay))(logit)
                

    return layers.Add()([conv_out, shortcut])

  def layer(stride, input_filter, output_filter):
    def func(logit):
       logit = WideResNet(input_filter, output_filter, stride)(logit)
       block_out = WideResNet(output_filter, output_filter, (1,1))(logit)
       return block_out
    return func

def create_model(channel_dim=2, k=2):
  input_vals = keras.Input(shape=(32, 32, 3))
  conv1_out = layers.Conv2D(16, (3, 3), padding='same')(input_vals)
  
  n_filters = [16, 16*k, 32*k, 64*k]

  #Residual Blocks
  conv2_out = WideResNet.layer((1, 1), n_filters[0], n_filters[1])(conv1_out)
  conv3_out = WideResNet.layer((2, 2), n_filters[1], n_filters[2])(conv2_out)  
  conv4_out = WideResNet.layer((2, 2), n_filters[2], n_filters[3])(conv3_out)

  logit = layers.BatchNormalization(axis=channel_dim)(conv4_out)
  logit = layers.Activation('relu')(logit)

  #Classifier
  out = layers.AveragePooling2D((8, 8), strides=(1, 1), padding='same')(logit)
  out = layers.Flatten()(out)
  output = layers.Dense(100, activation='softmax')(out)

  model = keras.Model(inputs=input_vals, outputs=output)

  return model

if __name__ == '__main__':
    K.clear_session()
    model = create_model()
    