from pyhdf.SD import SD, SDC
import h5py
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
from scipy.stats import t
from scipy.interpolate import UnivariateSpline
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import compute_class_weight
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
import tensorflow as tf
from keras.utils import to_categorical
from keras.src.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (ConvLSTM2D, Dense, Flatten, TimeDistributed, LSTM, Conv1D, Conv2D, MaxPooling2D,
                          GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU, ReLU, Multiply, Permute,
                          Reshape, Lambda, RepeatVector)
from keras import Input, Model
from keras.src.layers import Activation, Add, Concatenate, SpatialDropout2D, Bidirectional, AveragePooling2D, \
    SeparableConv2D, GRU
from keras.optimizers.legacy import Adam
from keras.src.optimizers import RMSprop
from keras.regularizers import l2
from keras_self_attention import SeqSelfAttention
from keras import backend as K
from keras.backend import sum as Ksum

from src.ForestHealthClassification import ForestHealthClassification


class ModelSelection:
    def __init__(self, sg_values, time_steps=14,
                 train_xy=(0, 800), train_hw=(2000, 2000),
                 val_xy=(0, 800), val_hw=(2000, 2000),
                 test_xy=(0, 2800), test_hw=(2000, 2000), scale=False):
        self.sg_values = np.where(sg_values[:, :, :] < 0, 0, sg_values[:, :, :])
        self.years = np.load('sg_trend_yearly_range.npy')
        self.scaler1 = StandardScaler()
        self.o = 1
        self.time_steps = time_steps
        self.train_xy, self.train_hw = train_xy, train_hw
        self.val_xy, self.val_hw = val_xy, val_hw
        self.test_xy, self.test_hw = test_xy, test_hw

        h1, w1 = train_xy
        h2, w2 = h1 + train_hw[0], w1 + train_hw[1]
        self.X_train = self.sg_values[0:time_steps, h1:h2, w1:w2].reshape(time_steps, -1).T
        if scale:
            self.X_train = self.scaler1.fit_transform(self.X_train)
        self.y_train = self.sg_values[time_steps, h1:h2, w1:w2].flatten()
        self.year_range_train = self.years[0:time_steps + 1]

        h1, w1 = test_xy
        h2, w2 = h1 + test_hw[0], w1 + test_hw[1]
        self.X_test = self.sg_values[self.o:time_steps + self.o, h1:h2, w1:w2].reshape(time_steps, -1).T
        if scale:
            self.X_test = self.scaler1.transform(self.X_test)
        self.y_test = self.sg_values[time_steps + self.o, h1:h2, w1:w2].flatten()
        self.r_test = self.years[self.o:time_steps + self.o + 1]
        self.d_test = self.sg_values[self.o:time_steps + self.o + 1, h1:h2, w1:w2]

        self.scaler2 = StandardScaler()
        h1, w1 = 600, 3600
        h2, w2 = h1 + 200, w1 + 200
        self.X_train_svr = self.sg_values[0:time_steps, h1:h2, w1:w2].reshape(time_steps, -1).T
        if scale:
            self.X_train_svr = self.scaler2.fit_transform(self.X_train_svr)
        self.y_train_svr = self.sg_values[time_steps, h1:h2, w1:w2].flatten()

        h1, w1 = test_xy
        h2, w2 = h1 + test_hw[0], w1 + test_hw[1]
        self.X_test_svr = self.sg_values[self.o:time_steps + self.o, h1:h2, w1:w2].reshape(time_steps, -1).T
        if scale:
            self.X_test_svr = self.scaler2.transform(self.X_test_svr)
        self.y_test_svr = self.sg_values[time_steps + self.o, h1:h2, w1:w2].flatten()

    def train_nn(self, idx, conv_filters, lstm_dropouts):
        h, w = self.train_hw

        if idx in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_gen1(h, w, scaling=False)
            if idx == 1:
                name, weight_file, model = self.nn1(idx, h, w, conv_filters, lstm_dropouts)
            elif idx == 2:
                name, weight_file, model = self.nn2(idx, h, w, conv_filters, lstm_dropouts)
            elif idx == 3:
                name, weight_file, model = self.nn3(h, w, conv_filters, lstm_dropouts)
            elif idx == 4:
                name, weight_file, model = self.nn4(h, w)
            elif idx == 5:
                name, weight_file, model = self.nn5(h, w)
            elif idx == 6:
                name, weight_file, model = self.nn1_v1(h, w)
            elif idx == 7:
                name, weight_file, model = self.nn1_v2(h, w)
            elif idx == 8:
                name, weight_file, model = self.nn1_v3(h, w)
            elif idx == 9:
                name, weight_file, model = self.nn1_v4(h, w)
            elif idx == 10:
                name, weight_file, model = self.nn1_v5(h, w)
            elif idx == 11:
                name, weight_file, model = self.nn1_v6(h, w)
            elif idx == 12:
                name, weight_file, model = self.nn1_v7(h, w)
            else:
                return
        else:
            return

        # Training from previous saved model
        try:
            model = load_model(weight_file, custom_objects={'SeqSelfAttention': SeqSelfAttention})
            print(f"Loaded model from {weight_file}. Continuing training.")
        except (ImportError, IOError):
            print(f"No saved model found: {weight_file}. Starting fresh training.")

        # early stop & check point
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=25),
            ModelCheckpoint(
                weight_file,
                monitor='val_loss',
                verbose=2,
                save_best_only=True,
                mode='min',
                save_freq='epoch'
            )
        ]

        # training
        history = model.fit(
            X_train, y_train,
            epochs=10000,
            batch_size=1,
            validation_data=(X_val, y_val),
            verbose=2,
            callbacks=callbacks
        )
        model.save(weight_file)
        self.training_performance_plot(history, weight_file.split(".", 1)[0])

        # eval
        model.load_weights(weight_file)
        loss = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}")

        # Predict and evaluate
        y_pred = model.predict(X_test)

        # Formatting and printing the results
        metrics = {
            'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
            'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
            'Root Mean Squared Error (RMSE)': mean_squared_error(y_test, y_pred, squared=False),
            'R-Squared (R2)': r2_score(y_test, y_pred)
        }

        print("LSTM Model Evaluation Metrics:")
        print("----------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("----------------------------")

        self.FH_clf_visualization(name, y_pred, metrics)

        return

    @staticmethod
    def train_test_data_dist(ds):
        plt.hist(ds.flatten(), bins=50)
        plt.title('Value Distribution')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.show()

    def train_test_gen1(self, h, w, scaling=False):
        scaler = StandardScaler()
        h1, w1 = self.train_xy
        h2, w2 = h1 + h, w1 + w
        X_train = self.sg_values[0:self.time_steps, h1:h2, w1:w2].reshape(self.time_steps, -1).T
        if scaling:
            X_train = scaler.fit_transform(X_train)
        X_train = X_train.reshape(-1, self.time_steps, h, w, 1)
        y_train = self.sg_values[self.time_steps, h1:h2, w1:w2].reshape(1, -1)

        h1, w1 = self.val_xy
        h2, w2 = h1 + h, w1 + w
        X_val = self.sg_values[1:self.time_steps + 1, h1:h2, w1:w2].reshape(self.time_steps, -1).T
        if scaling:
            X_val = scaler.transform(X_val)
        X_val = X_val.reshape(-1, self.time_steps, h, w, 1)
        y_val = self.sg_values[self.time_steps + 1, h1:h2, w1:w2].reshape(1, -1)

        h1, w1 = self.test_hw
        h2, w2 = h1 + h, w1 + w
        X_test = self.sg_values[1:self.time_steps + 1, h1:h2, w1:w2].reshape(self.time_steps, -1).T
        if scaling:
            X_test = scaler.transform(X_test)
        X_test = X_test.reshape(-1, self.time_steps, h, w, 1)
        y_test = self.sg_values[self.time_steps + 1, h1:h2, w1:w2].reshape(1, -1)

        print("X_train.shape = ", X_train.shape)
        print("y_train.shape = ", y_train.shape)
        print("X_val.shape = ", X_val.shape)
        print("y_val.shape = ", y_val.shape)
        print("X_test.shape = ", X_test.shape)
        print("y_test.shape = ", y_test.shape)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def nn1_v7(self, h, w):
        name = "LSTM_1_v7"
        weight_file = 'best_LSTM_model_1_v7.h5'

    def nn1_v6(self, h, w):
        """
        nn1_v1 + attention + Combine + SpatialDropout
        :param h:
        :param w:
        :return:
        """
        name = "LSTM_1_v6"
        weight_file = 'best_LSTM_model_1_v6.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(SpatialDropout2D(0.5))(x)

        # Second block with residual connection
        previous_block_activation = x
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = Add()([x, previous_block_activation])

        # Third block with Spatial Dropout
        x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(SpatialDropout2D(0.5))(x)

        # Global Average Pooling
        flat_conv_out = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        rnn_out = LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01))(flat_conv_out)
        rnn_out = Dropout(0.5)(rnn_out)
        # Compute the attention weights for each time step
        rnn_out = SeqSelfAttention(attention_activation='softmax')(rnn_out)
        rnn_out = LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.01))(rnn_out)
        rnn_out = Dropout(0.85)(rnn_out)

        # # Combine CNN and RNN outputs
        # combined = Concatenate()([Flatten()(flat_conv_out), rnn_out])
        #
        # # Dense layers
        # dense_out = Dropout(0.75)(rnn_out)

        # Output layer
        output = Dense(units=h * w, activation='linear')(rnn_out)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=100), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn1_v5(self, h, w):
        """
        nn1_v1 + attention + Combine + SpatialDropout
        :param h:
        :param w:
        :return:
        """
        name = "LSTM_1_v5"
        weight_file = 'best_LSTM_model_1_v5.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
        x = TimeDistributed(SpatialDropout2D(0.5))(x)

        # Second block with residual connection
        previous_block_activation = x
        for _ in range(2):
            x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                       kernel_regularizer=l2(0.001)))(x)
            x = TimeDistributed(BatchNormalization())(x)
            x = TimeDistributed(Activation('relu'))(x)
        x = Add()([x, previous_block_activation])

        # Global Average Pooling
        flat_conv_out = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        rnn_out = LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.01))(flat_conv_out)
        rnn_out = Dropout(0.5)(rnn_out)
        rnn_out = SeqSelfAttention(attention_activation='softmax')(rnn_out)
        rnn_out = LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.01))(rnn_out)
        rnn_out = Dropout(0.5)(rnn_out)

        # Combine CNN and RNN outputs
        combined = Concatenate()([Flatten()(flat_conv_out), rnn_out])

        # Dense layers
        dense_out = Dropout(0.75)(combined)

        # Output layer
        output = Dense(units=h * w, activation='linear')(dense_out)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=1), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn1_v4(self, h, w):
        """
        nn1_v1 + attention + Combine
        :param h:
        :param w:
        :return:
        """
        name = "LSTM_1_v4"
        weight_file = 'best_LSTM_model_1_v4.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction with residual connections
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(3, 3)))(x)

        # Second block with residual connection
        previous_block_activation = x
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Add()([x, previous_block_activation])

        x = TimeDistributed(Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(5, 5)))(x)

        # Global Average Pooling
        flat_conv_out = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        rnn_out = LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.01))(flat_conv_out)
        rnn_out = Dropout(0.1)(rnn_out)
        # Compute the attention weights for each time step
        rnn_out = SeqSelfAttention(attention_activation='softmax')(rnn_out)
        rnn_out = LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.01))(rnn_out)
        rnn_out = Dropout(0.1)(rnn_out)

        # Combine CNN and RNN outputs
        combined = Concatenate()([Flatten()(flat_conv_out), rnn_out])

        # Dense layers
        dense_out = Dropout(0.85)(combined)

        # Output layer
        output = Dense(units=h * w, activation='linear')(dense_out)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=1), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn1_v3(self, h, w):
        """
        nn1_v1 + attention + Combine
        :param h:
        :param w:
        :return:
        """
        name = "LSTM_1_v3"
        weight_file = 'best_LSTM_model_1_v3.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction with residual connections
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Second block with residual connection
        previous_block_activation = x
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = Add()([x, previous_block_activation])

        x = TimeDistributed(Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Global Average Pooling
        flat_conv_out = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        rnn_out = LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.01))(flat_conv_out)
        rnn_out = Dropout(0.1)(rnn_out)
        # Compute the attention weights for each time step
        rnn_out = SeqSelfAttention(attention_activation='softmax')(rnn_out)
        rnn_out = LSTM(units=32, return_sequences=False, kernel_regularizer=l2(0.01))(rnn_out)
        rnn_out = Dropout(0.1)(rnn_out)

        # Combine CNN and RNN outputs
        combined = Concatenate()([Flatten()(flat_conv_out), rnn_out])

        # Dense layers
        dense_out = Dropout(0.85)(combined)

        # Output layer
        output = Dense(units=h * w, activation='linear')(dense_out)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=0.2), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn1_v2(self, h, w):
        """
        nn1_v1 + attention
        :param h:
        :param w:
        :return:
        """
        name = "LSTM_1_v2"
        weight_file = 'best_LSTM_model_1_v2.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction with residual connections
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=l2(0.01)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Second block with residual connection
        previous_block_activation = x
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=l2(0.1)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=l2(0.1)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = Add()([x, previous_block_activation])

        # Global Average Pooling
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        x = LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.85)(x)
        # Compute the attention weights for each time step
        x = SeqSelfAttention(attention_activation='softmax')(x)
        x = LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.85)(x)

        # Output layer
        # x = Dense(units=32, activation='relu')(x)
        output = Dense(units=h * w, activation='linear')(x)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=100), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn1_v1(self, h, w):
        name = "LSTM_1_v1"
        weight_file = 'best_LSTM_model_1_v1.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Spatial feature extraction with residual connections
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=l2(10)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
        #                            kernel_regularizer=l2(10)))(x)
        # x = TimeDistributed(BatchNormalization())(x)
        # x = TimeDistributed(Activation('relu'))(x)
        # x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=l2(10)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Global Average Pooling
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # Temporal feature extraction with LSTM and attention
        x = LSTM(units=64, return_sequences=True, kernel_regularizer=l2(100))(x)
        x = Dropout(0.8)(x)
        x = LSTM(units=64, return_sequences=False, kernel_regularizer=l2(100))(x)
        x = Dropout(0.8)(x)

        # Output layer
        # x = Dense(units=32, activation='relu')(x)
        output = Dense(units=h * w, activation='linear')(x)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=100), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    @staticmethod
    def attention_block(x, dim, activation='tanh', name='attention_vector'):
        attention = TimeDistributed(Dense(1, activation=activation))(x)
        attention = Permute((2, 1))(attention)
        attention = Lambda(lambda xx: K.sum(xx, axis=-1), name=name)(attention)
        attention = RepeatVector(dim)(attention)
        attention = Permute((2, 1))(attention)
        x = Multiply()([x, attention])
        return x

    def nn5(self, h, w, conv_filters, convlstm_units, l2_reg=0.001):
        dropout_str = "-".join(["drops"] + [f"{d:.2f}".replace('.', '_') for d in convlstm_units])
        name = f"LSTM5__{dropout_str}"
        weight_file = f'Best_LSTM5__{dropout_str}.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        # Inception Module
        tower_1 = TimeDistributed(Conv2D(conv_filters[0], (1, 1), padding='same', activation='relu'))(input_layer)
        tower_2 = TimeDistributed(Conv2D(conv_filters[1], (3, 3), padding='same', activation='relu'))(input_layer)
        tower_3 = TimeDistributed(Conv2D(conv_filters[2], (5, 5), padding='same', activation='relu'))(input_layer)
        x = Concatenate()([tower_1, tower_2, tower_3])

        # Spatial Pyramid Pooling
        x = TimeDistributed(SpatialPyramidPooling([1, 2, 4]))(x)

        # ConvLSTM Layers
        x = ConvLSTM2D(convlstm_units[0], (3, 3), padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = ConvLSTM2D(convlstm_units[1], (3, 3), padding='same', return_sequences=False)(x)
        x = BatchNormalization()(x)

        # Dense Layers
        x = Flatten()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(0.4)(x)

        # Output Layer
        output = Dense(units=h * w, activation='linear')(x)

        # Compile Model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn4(self, h, w, conv_filters, gru_units, gru_dropouts, l2_reg=0.001):
        dropout_str = "-".join(["drops"] + [f"{d:.2f}".replace('.', '_') for d in gru_dropouts])
        name = f"LSTM4__{dropout_str}"
        weight_file = f'Best_LSTM4__{dropout_str}.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        f1, f2, f3 = conv_filters
        u1, u2, u3 = gru_units
        d1, d2, d3 = gru_dropouts

        # First Conv Block with Depthwise Separable Convolution
        x = TimeDistributed(SeparableConv2D(filters=f1, kernel_size=(3, 3), padding='same',
                                            dilation_rate=1, depth_multiplier=2,
                                            kernel_regularizer=l2(l2_reg)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Multi-Scale Convolution
        conv_3x3 = TimeDistributed(Conv2D(filters=f2, kernel_size=(3, 3), padding='same'))(x)
        conv_5x5 = TimeDistributed(Conv2D(filters=f2, kernel_size=(5, 5), padding='same'))(x)
        x = Concatenate()([conv_3x3, conv_5x5])
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)

        # Dilated Convolution Block
        x = TimeDistributed(Conv2D(filters=f3, kernel_size=(3, 3), padding='same', dilation_rate=2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Global Average Pooling
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # GRU Layers with Attention
        x = GRU(units=u1, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d1)(x)
        x = SeqSelfAttention(attention_activation='softmax')(x)
        x = GRU(units=u2, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d2)(x)
        x = GRU(units=u3, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d3)(x)

        # Dense Output Layer
        x = Dense(units=128, activation='relu')(x)
        output = Dense(units=h * w, activation='linear')(x)

        # Compile Model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    def nn3(self, idx, h, w, conv_filters, lstm_dropouts, l2_reg=0.001):

        return name, weight_file, model

    def nn2(self, idx, h, w, conv_filters, lstm_dropouts):
        dropout_str = "-".join(["drops"] + [f"{d:.2f}".replace('.', '_') for d in gru_dropouts])
        name = f"LSTM4__{dropout_str}"
        weight_file = f'Best_LSTM4__{dropout_str}.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        f1, f2, f3 = conv_filters
        u1, u2, u3 = gru_units
        d1, d2, d3 = gru_dropouts

        # First Conv Block with Depthwise Separable Convolution
        x = TimeDistributed(SeparableConv2D(filters=f1, kernel_size=(3, 3), padding='same',
                                            dilation_rate=1, depth_multiplier=2,
                                            kernel_regularizer=l2(l2_reg)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Multi-Scale Convolution
        conv_3x3 = TimeDistributed(Conv2D(filters=f2, kernel_size=(3, 3), padding='same'))(x)
        conv_5x5 = TimeDistributed(Conv2D(filters=f2, kernel_size=(5, 5), padding='same'))(x)
        x = Concatenate()([conv_3x3, conv_5x5])
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)

        # Dilated Convolution Block
        x = TimeDistributed(Conv2D(filters=f3, kernel_size=(3, 3), padding='same', dilation_rate=2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

        # Global Average Pooling
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        # GRU Layers with Attention
        x = GRU(units=u1, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d1)(x)
        x = SeqSelfAttention(attention_activation='softmax')(x)
        x = GRU(units=u2, return_sequences=True, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d2)(x)
        x = GRU(units=u3, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(d3)(x)

        # Dense Output Layer
        x = Dense(units=128, activation='relu')(x)
        output = Dense(units=h * w, activation='linear')(x)

        # Compile Model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error', metrics=['mae'])
        model.summary()
        
        return name, weight_file, model

    def nn1(self, id, h, w, conv_filters, lstm_dropouts):
        dropout_str = "-".join(["drops"] + [f"{d:.2f}".replace('.', '_') for d in lstm_dropouts])
        name = f"LSTM{id}__{dropout_str}"
        weight_file = f'Best_LSTM{id}__{dropout_str}.h5'

        input_layer = Input(shape=(self.time_steps, h, w, 1))

        f1, f2, f3 = conv_filters
        x = TimeDistributed(Conv2D(filters=f1, kernel_size=(3, 3), padding='same', dilation_rate=2,
                                   kernel_regularizer=l2(l2_reg)))(input_layer)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(3, 3)))(x)

        previous_block_activation = x
        for _ in range(2):
            x = TimeDistributed(Conv2D(filters=f2, kernel_size=(5, 5), padding='same', activation='relu',
                                       kernel_regularizer=l2(l2_reg)))(x)
            x = TimeDistributed(BatchNormalization())(x)
            x = TimeDistributed(Activation('relu'))(x)
        x = Add()([x, previous_block_activation])

        x = TimeDistributed(Conv2D(filters=f3, kernel_size=(7, 7), padding='same', dilation_rate=2,
                                   kernel_regularizer=l2(l2_reg)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(5, 5)))(x)

        # Global Average Pooling
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        d1, d2 = lstm_dropouts
        x = LSTM(units=128, activation='relu', kernel_regularizer=l2(l2_reg), return_sequences=True)(x)
        x = Dropout(d1)(x)
        x = SeqSelfAttention(attention_activation='softmax')(x)
        x = LSTM(units=64, activation='relu', kernel_regularizer=l2(l2_reg), return_sequences=True)(x)
        x = Dropout(d2)(x)

        # Output layer
        x = Dense(units=128, activation='relu')(x)
        output = Dense(units=h * w, activation='linear')(x)

        # Compile model
        model = Model(inputs=input_layer, outputs=output, name=name)
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
        model.summary()

        return name, weight_file, model

    @staticmethod
    def training_performance_plot(history, weight_file):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Performance')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        # plt.show()
        plt.savefig(f"{weight_file}.png")
        plt.close(fig)
        return

    def train_lr(self):
        """
        ----------------------------
        Linear Regression Model Evaluation Metrics:
        ----------------------------
        ----------------------------
        Mean Absolute Error (MAE): 1854.7304
        Mean Squared Error (MSE): 6865146.5055
        Root Mean Squared Error (RMSE): 2620.1425
        R-Squared (R2): 0.8996
        ----------------------------
        :return:
        """
        name = "Linear Regression"
        self.__print_header(name)
        model = LinearRegression()
        y_pred, metrics = self.__train(model)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, model, y_pred, metrics

    def train_xgb(self):
        """
        ----------------------------
        XGBoost Model Evaluation Metrics:
        ----------------------------
        Fitting 5 folds for each of 10 candidates, totalling 50 fits
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.7; total time=   4.0s
        [CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=35, subsample=1; total time=   3.0s
        [CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=35, subsample=1; total time=   3.0s
        [CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=35, subsample=1; total time=   3.0s
        [CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=35, subsample=1; total time=   3.0s
        [CV] END colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=35, subsample=1; total time=   2.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.7s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   3.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   5.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   5.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   5.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   5.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=1; total time=   4.9s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.8s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.9s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.7; total time=   3.8s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.1s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=3, n_estimators=15, subsample=0.7; total time=   0.8s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=3, n_estimators=15, subsample=0.7; total time=   0.8s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=3, n_estimators=15, subsample=0.7; total time=   0.8s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=3, n_estimators=15, subsample=0.7; total time=   0.8s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=3, n_estimators=15, subsample=0.7; total time=   0.8s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   7.7s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   7.8s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   7.7s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   7.7s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=5, n_estimators=100, subsample=0.7; total time=   7.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=35, subsample=1; total time=   1.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=35, subsample=1; total time=   1.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=35, subsample=1; total time=   1.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=35, subsample=1; total time=   1.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=3, n_estimators=35, subsample=1; total time=   1.0s
        ----------------------------
        Mean Absolute Error (MAE): 2454.4694
        Mean Squared Error (MSE): 10717081.0328
        Root Mean Squared Error (RMSE): 3273.6953
        R-Squared (R2): 0.8433
        ----------------------------
        model best params:  {'subsample': 0.7, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.8}
        :return:
        """
        name = "XGBoost"
        self.__print_header(name)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
        param_grid = {
            'n_estimators': [15, 35, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 1],
            'colsample_bytree': [0.7, 0.8, 1]
        }
        model = RandomizedSearchCV(model, param_grid, cv=5, verbose=2)
        y_pred, metrics = self.__train(model)
        print("model best params: ", model.best_params_)
        self.FH_clf(name, y_pred, metrics)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, model, y_pred, metrics

    def train_dt(self):
        """
        ----------------------------
        Decision Tree Model Evaluation Metrics:
        ----------------------------
        Fitting 5 folds for each of 10 candidates, totalling 50 fits
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.7s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.9s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.3s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  15.0s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.0s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.0s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.1s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10; total time=  14.7s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10; total time=  15.1s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10; total time=  14.8s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10; total time=  15.0s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10; total time=  15.2s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10; total time=  12.0s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.7s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  13.7s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.1s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  13.8s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.1s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.2s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        ----------------------------
        Mean Absolute Error (MAE): 2478.5676
        Mean Squared Error (MSE): 10991524.0587
        Root Mean Squared Error (RMSE): 3315.3467
        R-Squared (R2): 0.8393
        ----------------------------
        model best params:  {'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 10}
        :return:
        """
        name = 'Decision Tree'
        self.__print_header(name)
        model = DecisionTreeRegressor()
        param_grid = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4]
        }
        model = RandomizedSearchCV(model, param_grid, cv=5, verbose=2)
        y_pred, metrics = self.__train(model)
        print("model best params: ", model.best_params_)
        self.FH_clf(name, y_pred, metrics)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, model, y_pred, metrics

    def train_svr(self):
        name = "SVM Regression"
        self.__print_header(name)
        model = SVR(kernel='linear')
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
        model = RandomizedSearchCV(model, param_grid, cv=5, verbose=2)
        model.fit(self.X_train_svr, self.y_train_svr)
        y_pred = model.predict(self.X_test_svr)
        metrics = self.__model_eval(y_pred, self.y_test_svr)
        print("model best params: ", model.best_params_)
        self.FH_clf(name, y_pred, metrics)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, model, y_pred, metrics

    def train_rf(self):
        """
        ----------------------------
        Random Forest Model Evaluation Metrics:
        ----------------------------
        Fitting 5 folds for each of 10 candidates, totalling 50 fits
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=45; total time= 5.9min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=45; total time= 6.0min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=45; total time= 6.1min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=45; total time= 6.2min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=45; total time= 7.0min
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 5.2min
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 5.3min
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 5.3min
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 5.5min
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 6.2min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=15; total time= 2.0min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=15; total time= 2.0min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=15; total time= 2.0min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=15; total time= 2.1min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=15; total time= 2.3min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.6min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.8min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 5.7min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 5.8min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 5.8min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 5.9min
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 6.7min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time=  52.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time=  52.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time=  52.7s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time=  55.2s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=15; total time= 1.0min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=15; total time= 1.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=15; total time= 1.6min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=15; total time= 1.8min
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 4.6min
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 4.6min
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 4.6min
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 4.8min
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=45; total time= 5.5min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 2.6min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 2.6min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 2.6min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 2.8min
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=45; total time= 3.1min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=45; total time= 4.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=45; total time= 4.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=45; total time= 4.5min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=45; total time= 4.7min
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2, n_estimators=45; total time= 5.4min
        ----------------------------
        Mean Absolute Error (MAE): 1781.9302
        Mean Squared Error (MSE): 6276843.6926
        Root Mean Squared Error (RMSE): 2505.3630
        R-Squared (R2): 0.9647
        ----------------------------
        :return:
        """
        name = 'Random Forest'
        self.__print_header(name)
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [15, 45],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4]
        }
        model = RandomizedSearchCV(model, param_grid, cv=5, verbose=2)
        y_pred, metrics = self.__train(model)
        print("model best params: ", model.best_params_)
        self.FH_clf(name, y_pred, metrics)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, model, y_pred, metrics

    def train_ensemble_model(self, models):
        """
        ----------------------------
        Ensemble Model Model Evaluation Metrics:
        ----------------------------
        Fitting 5 folds for each of 10 candidates, totalling 50 fits
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=15, subsample=0.8; total time=   1.5s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=15, subsample=0.8; total time=   1.4s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=15, subsample=0.8; total time=   1.4s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=15, subsample=0.8; total time=   1.4s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=7, n_estimators=15, subsample=0.8; total time=   1.4s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.7; total time=   3.4s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.7; total time=   3.4s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.7; total time=   3.4s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.7; total time=   3.4s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.7; total time=   3.4s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=7, n_estimators=35, subsample=1; total time=   3.6s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=7, n_estimators=35, subsample=1; total time=   3.6s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=7, n_estimators=35, subsample=1; total time=   3.6s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=7, n_estimators=35, subsample=1; total time=   3.6s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=7, n_estimators=35, subsample=1; total time=   3.5s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=15, subsample=0.8; total time=   0.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=15, subsample=0.8; total time=   0.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=15, subsample=0.8; total time=   0.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=15, subsample=0.8; total time=   0.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.01, max_depth=3, n_estimators=15, subsample=0.8; total time=   0.6s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   5.8s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   5.9s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   6.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   6.0s
        [CV] END colsample_bytree=0.7, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   5.9s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.7s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.7s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.6s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.7s
        [CV] END colsample_bytree=1, learning_rate=0.2, max_depth=5, n_estimators=35, subsample=0.8; total time=   2.7s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=15, subsample=1; total time=   0.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=15, subsample=1; total time=   0.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=15, subsample=1; total time=   0.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=15, subsample=1; total time=   0.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=15, subsample=1; total time=   0.9s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=3, n_estimators=35, subsample=1; total time=   1.3s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=3, n_estimators=35, subsample=1; total time=   1.3s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=3, n_estimators=35, subsample=1; total time=   1.3s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=3, n_estimators=35, subsample=1; total time=   1.3s
        [CV] END colsample_bytree=1, learning_rate=0.01, max_depth=3, n_estimators=35, subsample=1; total time=   1.3s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.8; total time=   3.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.8; total time=   3.2s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.8; total time=   3.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.8; total time=   3.1s
        [CV] END colsample_bytree=0.7, learning_rate=0.2, max_depth=7, n_estimators=35, subsample=0.8; total time=   3.1s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1; total time=   7.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1; total time=   7.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1; total time=   7.9s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1; total time=   7.8s
        [CV] END colsample_bytree=0.8, learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1; total time=   8.0s
        Fitting 5 folds for each of 10 candidates, totalling 50 fits
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  13.7s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.1s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  13.8s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.0s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=10; total time=  14.2s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2; total time=  12.0s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  13.8s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.3s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.3s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.4s
        [CV] END max_depth=None, min_samples_leaf=4, min_samples_split=2; total time=  14.7s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2; total time=  15.8s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2; total time=  16.2s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2; total time=  15.8s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2; total time=  16.2s
        [CV] END max_depth=None, min_samples_leaf=1, min_samples_split=2; total time=  16.4s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.9s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=2; total time=  11.8s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=10; total time=   6.5s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10; total time=  11.6s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10; total time=  11.8s
        [CV] END max_depth=20, min_samples_leaf=4, min_samples_split=10; total time=  11.8s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=4, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2; total time=   6.5s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.6s
        [CV] END max_depth=10, min_samples_leaf=1, min_samples_split=10; total time=   6.5s
        ----------------------------
        Mean Absolute Error (MAE): 2071.2736
        Mean Squared Error (MSE): 7911453.9136
        Root Mean Squared Error (RMSE): 2812.7307
        R-Squared (R2): 0.8844
        ----------------------------
        :param models:
        :return:
        """
        name = 'Ensemble Model'
        self.__print_header(name)
        ensemble = VotingRegressor(models)
        y_pred, metrics = self.__train(ensemble)
        self.FH_clf(name, y_pred, metrics)
        self.FH_clf_visualization(name, y_pred, metrics)
        return name, ensemble, y_pred, metrics

    def __train(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        metrics = self.__model_eval(y_pred, self.y_test)
        return y_pred, metrics

    @staticmethod
    def __print_header(name):
        print("----------------------------")
        print(name, "Model Evaluation Metrics:")
        print("----------------------------")

    @staticmethod
    def __model_eval(y_pred, y_test):
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Formatting and printing the results
        metrics = {
            'Mean Absolute Error (MAE)': mae,
            'Mean Squared Error (MSE)': mse,
            'Root Mean Squared Error (RMSE)': rmse,
            'R-Squared (R2)': r2
        }

        print("----------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("----------------------------")

        return metrics

    @staticmethod
    def print_run_time(start_time):
        hours, remainder = divmod((datetime.now() - start_time).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"run time: {hours:02}:{minutes:02}:{seconds:02}")
        return

    def FH_clf(self, clf_name, y_pred, metrics, lr_range=10):
        d_pred = self.d_test.copy()
        d_pred[-1] = y_pred.reshape(self.test_hw[0], self.test_hw[1])
        clf = ForestHealthClassification(clf_name, d_pred, metrics, self.r_test, lr_range=lr_range)
        clf.linear_regress()
        clf.classify_pixels()
        return clf

    def FH_clf_visualization(self, name, y_pred, metrics):
        clf_gt = self.FH_clf("Ground Truth", self.y_test, None)
        fig1 = clf_gt.plot_it()

        clf_pred = self.FH_clf(name, y_pred, metrics)
        fig2 = clf_pred.plot_it()

        fig3 = self.FH_clf_acc(name, clf_pred, clf_pred.pixel_classes.flatten(), clf_gt.pixel_classes.flatten())

        # Create a single figure with subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(self.convert_fig_to_image(fig1))
        axs[1].imshow(self.convert_fig_to_image(fig2))
        axs[2].imshow(self.convert_fig_to_image(fig3))

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_visible(False)  # remove the spines

        fig.suptitle(f'Forest Health Prediction and Accuracy ({clf_pred.clf_name})', fontsize=12, y=0.95)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=0, hspace=0)
        plt.tight_layout(pad=-1)

        pth = f"{clf_pred.rootpath}/predicting/{clf_pred.lr_range}_years_range"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"{pth}/Forest_Health_Prediction_{clf_pred.year_range}({clf_pred.clf_name}).png"
        plt.savefig(fp)
        print(f"plot saved at {fp}")

        # plt.show()

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig)

        return

    @staticmethod
    def FH_clf_acc(name, clf, cls_pred, cls_truth):
        cm = confusion_matrix(cls_truth, cls_pred)

        log_cm = np.log(cm + 1)

        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(log_cm, cmap='Blues')

        fig.colorbar(cax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=15,  # Smaller font size
                        color='black' if log_cm[i, j] > 0.5 * np.max(log_cm) else 'black')

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Forest Health Classification Accuracy ({name})')

        pth = f"{clf.rootpath}/predicting/{clf.lr_range}_years_range"
        if not os.path.exists(pth):
            os.makedirs(pth)
        fp = f"{pth}/Forest_Health_Classification_Accuracy_{clf.year_range}({clf.clf_name}).png"
        plt.savefig(fp)
        print(f"plot saved at {fp}")

        return fig

    @staticmethod
    def convert_fig_to_image(fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image


if __name__ == '__main__':
    sg_h5 = f'./sg.h5'
    if os.path.exists(sg_h5):
        print(f"Read SG values from {sg_h5}")
        t1 = datetime.now()
        with h5py.File(sg_h5, 'r') as hf:
            arr = hf['SG'][:]
        ModelSelection.print_run_time(t1)
    else:
        print("File not found: ", sg_h5)
        exit(123)

    ms = ModelSelection(arr)

    name_lr, model_lr, y_pred_lr, metrics_lr = ms.train_lr()
    name_xgb, model_xgb, y_pred_xgb, metrics_xgb = ms.train_xgb()
    # name_rf, model_rf, y_pred_rf, metrics_rf = ms.train_rf()
    name_dt, model_dt, y_pred_dt, metrics_dt = ms.train_dt()
    # name_svr, model_svr, y_pred_svr, metrics_svr = ms.train_svr()
    name_en, model_en, y_pred_en, metrics_en = ms.train_ensemble_model([(name_lr, model_lr),
                                                                        (name_xgb, model_xgb),
                                                                        # (name_rf, model_rf),
                                                                        (name_dt, model_dt),
                                                                        # (name_svr, model_svr)
                                                                        ])
