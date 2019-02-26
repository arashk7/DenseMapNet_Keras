from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Input, Concatenate, Conv2DTranspose
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras.optimizers import *

class DenseMapNet:
    def __init__(self):
        super().__init__()
        self.model = None

    # Initialize the model
    def init_model(self):
        Input_1 = Input(shape=(100, 150, 1), name='Input_1')
        Input_2 = Input(shape=(100, 150, 1), name='Input_2')
        merge_1 = Concatenate()([Input_1, Input_2])
        Convolution2D_21 = Convolution2D(name='Convolution2D_21', nb_col=5, nb_filter=1, border_mode='same', nb_row=5)(
            Input_2)
        Convolution2D_11 = Convolution2D(name='Convolution2D_11', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Input_2)
        MaxPooling2D_3 = MaxPooling2D(name='MaxPooling2D_3', pool_size=(8, 8))(Convolution2D_11)
        Convolution2D_1 = Convolution2D(name='Convolution2D_1', nb_col=5, nb_filter=32, border_mode='same', nb_row=5)(
            merge_1)
        MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1', pool_size=(8, 8))(Convolution2D_1)
        BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(MaxPooling2D_1)
        Activation_1 = Activation(name='Activation_1', activation='relu')(BatchNormalization_1)
        Convolution2D_2 = Convolution2D(name='Convolution2D_2', nb_col=5, nb_filter=32, border_mode='same', nb_row=5)(
            Activation_1)
        Dropout_1 = Dropout(name='Dropout_1', p=0.2)(Convolution2D_2)
        Convolution2D_4 = Convolution2D(name='Convolution2D_4', nb_col=5, nb_filter=32, border_mode='same', nb_row=5)(
            Activation_1)
        Dropout_3 = Dropout(name='Dropout_3', p=0.2)(Convolution2D_4)
        Convolution2D_5 = Convolution2D(name='Convolution2D_5', nb_col=5, nb_filter=32, border_mode='same', nb_row=5)(
            Activation_1)
        Dropout_4 = Dropout(name='Dropout_4', p=0.2)(Convolution2D_5)
        Convolution2D_3 = Convolution2D(name='Convolution2D_3', nb_col=5, nb_filter=32, border_mode='same', nb_row=5)(
            Activation_1)
        Dropout_2 = Dropout(name='Dropout_2', p=0.2)(Convolution2D_3)
        merge_2 = Concatenate()([Dropout_1, Dropout_2, Activation_1, Dropout_3, Dropout_4])
        merge_5 = Concatenate()([MaxPooling2D_3, merge_2])
        BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3')(merge_5)
        Activation_3 = Activation(name='Activation_3', activation='relu')(BatchNormalization_3)
        Convolution2D_12 = Convolution2D(name='Convolution2D_12', nb_col=1, nb_filter=64, border_mode='same', nb_row=1)(
            Activation_3)
        BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4')(Convolution2D_12)
        Activation_4 = Activation(name='Activation_4', activation='relu')(BatchNormalization_4)
        Convolution2D_13 = Convolution2D(name='Convolution2D_13', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Activation_4)
        Dropout_9 = Dropout(name='Dropout_9', p=0.2)(Convolution2D_13)
        merge_6 = Concatenate()([Dropout_9, merge_5])
        BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5')(merge_6)
        Activation_5 = Activation(name='Activation_5', activation='relu')(BatchNormalization_5)
        Convolution2D_14 = Convolution2D(name='Convolution2D_14', nb_col=1, nb_filter=64, border_mode='same', nb_row=1)(
            Activation_5)
        BatchNormalization_6 = BatchNormalization(name='BatchNormalization_6')(Convolution2D_14)
        Activation_6 = Activation(name='Activation_6', activation='relu')(BatchNormalization_6)
        Convolution2D_15 = Convolution2D(name='Convolution2D_15', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Activation_6)
        Dropout_10 = Dropout(name='Dropout_10', p=0.2)(Convolution2D_15)
        merge_7 = Concatenate()([Dropout_10, merge_6])
        BatchNormalization_7 = BatchNormalization(name='BatchNormalization_7')(merge_7)
        Activation_7 = Activation(name='Activation_7', activation='relu')(BatchNormalization_7)
        Convolution2D_16 = Convolution2D(name='Convolution2D_16', nb_col=1, nb_filter=64, border_mode='same', nb_row=1)(
            Activation_7)
        BatchNormalization_8 = BatchNormalization(name='BatchNormalization_8')(Convolution2D_16)
        Activation_8 = Activation(name='Activation_8', activation='relu')(BatchNormalization_8)
        Convolution2D_17 = Convolution2D(name='Convolution2D_17', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Activation_8)
        Dropout_11 = Dropout(name='Dropout_11', p=0.2)(Convolution2D_17)
        merge_8 = Concatenate()([Dropout_11, merge_7])
        BatchNormalization_9 = BatchNormalization(name='BatchNormalization_9')(merge_8)
        Activation_9 = Activation(name='Activation_9', activation='relu')(BatchNormalization_9)
        Convolution2D_18 = Convolution2D(name='Convolution2D_18', nb_col=1, nb_filter=64, border_mode='same', nb_row=1)(
            Activation_9)
        BatchNormalization_10 = BatchNormalization(name='BatchNormalization_10')(Convolution2D_18)
        Activation_10 = Activation(name='Activation_10', activation='relu')(BatchNormalization_10)
        Convolution2D_19 = Convolution2D(name='Convolution2D_19', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Activation_10)
        Dropout_12 = Dropout(name='Dropout_12', p=0.2)(Convolution2D_19)
        merge_9 = Concatenate()([Dropout_12, merge_8])
        BatchNormalization_11 = BatchNormalization(name='BatchNormalization_11')(merge_9)
        Activation_11 = Activation(name='Activation_11', activation='relu')(BatchNormalization_11)
        Convolution2D_20 = Convolution2D(name='Convolution2D_20', nb_col=1, nb_filter=32, border_mode='same', nb_row=1)(
            Activation_11)
        UpSampling2D_1 = UpSampling2D(name='UpSampling2D_1', size=(8, 8))(Convolution2D_20)
        ZeroPadding2D_1 = ZeroPadding2D(name='ZeroPadding2D_1', padding=(2, 3))(UpSampling2D_1)
        merge_10 = Concatenate(axis=3)([ZeroPadding2D_1, Convolution2D_21])
        BatchNormalization_12 = BatchNormalization(name='BatchNormalization_12')(merge_10)
        Activation_12 = Activation(name='Activation_12', activation='relu')(BatchNormalization_12)
        Convolution2D_22 = Convolution2D(name='Convolution2D_22', nb_col=5, nb_filter=16, border_mode='same', nb_row=5)(
            Activation_12)
        merge_11 = Concatenate(axis=3)([Convolution2D_22, merge_10])
        BatchNormalization_13 = BatchNormalization(name='BatchNormalization_13')(merge_11)
        Activation_13 = Activation(name='Activation_13', activation='relu')(BatchNormalization_13)
        Convolution2D_23 = Conv2DTranspose(name='Convolution2D_23', nb_col=9, nb_filter=1, border_mode='same',
                                           nb_row=9)(
            Activation_13)
        Activation_14 = Activation(name='Activation_14', activation='sigmoid')(Convolution2D_23)

        model = Model([Input_1, Input_2], [Activation_14])

        print(model.summary())
        self.model = model
        return model

    # Return the model optimizer
    def get_optimizer(self):
        return RMSprop(lr=1e-3)

    # Return the model Loss function
    def get_loss_function(self):
        return 'mean_squared_error'

    # Return the Batch size
    def get_batch_size(self):
        return 32

    # Return the default number of epochs
    def get_num_epoch(self):
        return 100

    # Load model and weights from disk
    def load_model_and_weight(self, model_name):
        # load model
        json_file = open(model_name + '.json', 'r')
        model = json_file.read()
        json_file.close()
        model = model_from_json(model)
        # load weights into model
        model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        self.model = model

    # Save model and weights into model directory
    def save_model_and_weight(self, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + '.h5')
        print("Saved model to disk")

    # Compile the model
    def compile(self):
        self.model.compile(optimizer=self.get_optimizer(), loss=self.get_loss_function(), metrics=['accuracy'])
        return self.model

    # Train the model
    def train(self, x, y, n_epoch=20, batch_size=1):
        self.model.fit(x, y, epochs=n_epoch, batch_size=batch_size, verbose=1)

    # Check the error rate on its input test data (x_test & y_test) and print the result in consule
    def get_error_rate(self, x_ts, y_ts):
        p = self.model.predict(x_ts, batch_size=self.get_batch_size(), verbose=0)
        mse = np.mean(np.square(y_ts - p))
        print("Error rate is " + str(mse))
        return mse
