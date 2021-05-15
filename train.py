import model
from preprocessing import get_data
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from logger import App_Logger



def train():
    

    strategy = tf.distribute.MirroredStrategy()

    (x_train, y_train), (x_test, y_test) = get_data() 
    with open('DataLoading.txt', 'a+') as f:
        App_Logger.log(f, 'Loaded data successfully...')


    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs'), 
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1),
                    keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.01, verbose=1)]


    try:
        with strategy.scope():
        
            K.clear_session()
            myModel = model.create_model()
            myModel.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                                                                      optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
            
        with open('DataLoading.txt', 'a+') as f:
            App_Logger.log(f, 'Created and compiled model....\n' + myModel.summary()) 
    
        history = myModel.fit(x_train, y_train, validation_split=0.25, callbacks=callbacks, verbose=1)
        
        
        with open('train.txt', 'a+') as f:
            App_Logger.log(f, 'Training successful ' + history.history)
    except Exception as e:
        with open('Error.txt', 'a+') as f:
            App_Logger.log(f, e)

if __name__ == '__main__':
    train()

