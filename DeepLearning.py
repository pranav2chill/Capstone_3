import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  # For performing time series cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For standardizing numerical data and encoding categorical data
from tensorflow.keras.models import Sequential  # For building a linear stack of layers in the neural network
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, Conv1D, MaxPooling1D, Flatten  # Various types of layers that can be added to the neural network
from tensorflow.keras.optimizers import Adam  # The optimizer used to update network weights iterative based in training data
from tensorflow.keras.callbacks import EarlyStopping  # To stop training when a monitored metric has stopped improving
from tensorflow.keras.regularizers import l2  # L2 regularizer, used to avoid overfitting
from tensorflow.keras.metrics import Precision, Recall  # Metrics to measure the performance of the model

class DeepLearning:
    def __init__(self, data):
        """
        Initialize the DeepLearning class.

        Parameters:
        - data: pandas DataFrame, the input data for deep learning.
        """
        self.data = data

    def preprocess_data(self):
        """
        Preprocess the data for deep learning.

        Returns:
        - X_scaled: numpy array, the preprocessed input features.
        - y: numpy array, the encoded target variable.
        """
        numerical_features = ['adr', 'kill', 'death', 'assist', 'fk', 'fd', 'acs']
        for feature in numerical_features:
            self.data[feature] = self.data[feature].apply(lambda x: x.replace('\xa0', '') if isinstance(x, str) else x).replace('', np.nan).astype(float)
        
        # Fill NaN values with the mean (or median) of each column
        self.data[numerical_features] = self.data[numerical_features].fillna(self.data[numerical_features].mean())
        
        scaler = StandardScaler()
        # Scale numerical features
        X_scaled = scaler.fit_transform(self.data[numerical_features])
        
        # Encoding target variable
        y = np.where(self.data['win_lose'] == 'team win', 1, 0)

        # Assuming 'map' and 'map_pick' are categorical and need to be encoded
        encoder = OneHotEncoder(sparse=False)
        categorical_features = ['map', 'agent', 'map_pick']
        if 'map' in self.data.columns and 'map_pick' in self.data.columns:
            categorical_encoded = encoder.fit_transform(self.data[categorical_features])
            X_scaled = np.hstack((X_scaled, categorical_encoded))

        return X_scaled, y

    def build_model(self, input_shape):
        """
        Build a multi-layer perceptron (MLP) model.

        Parameters:
        - input_shape: int, the shape of the input features.

        Returns:
        - model: keras Sequential model, the built MLP model.
        """
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        return model

    def build_lstm_model(self, input_shape):
        """
        Build a Long Short-Term Memory (LSTM) model.

        Parameters:
        - input_shape: tuple, the shape of the input features.

        Returns:
        - model: keras Sequential model, the built LSTM model.
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            LSTM(32, kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        return model

    def build_simple_rnn_model(self, input_shape):
        """
        Build a Simple Recurrent Neural Network (RNN) model.

        Parameters:
        - input_shape: tuple, the shape of the input features.

        Returns:
        - model: keras Sequential model, the built Simple RNN model.
        """
        model = Sequential([
            SimpleRNN(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            SimpleRNN(32, kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        return model

    def build_cnn_model(self, input_shape):
        """
        Build a Convolutional Neural Network (CNN) model.

        Parameters:
        - input_shape: tuple, the shape of the input features.

        Returns:
        - model: keras Sequential model, the built CNN model.
        """
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
        return model

    def cross_validate_cnn(self, X, y, epochs=20, n_splits=5):
        """
        Perform cross-validation using a CNN model.

        Parameters:
        - X: numpy array, the input features.
        - y: numpy array, the target variable.
        - epochs: int, the number of epochs for training.
        - n_splits: int, the number of splits for cross-validation.
        """
        self.cross_validate_general(X, y, epochs, n_splits, model_type='cnn')

    def cross_validate_simple_rnn(self, X, y, epochs=20, n_splits=5):
        """
        Perform cross-validation using a Simple RNN model.

        Parameters:
        - X: numpy array, the input features.
        - y: numpy array, the target variable.
        - epochs: int, the number of epochs for training.
        - n_splits: int, the number of splits for cross-validation.
        """
        self.cross_validate_general(X, y, epochs, n_splits, model_type='simple_rnn')

    def cross_validate_general(self, X, y, epochs, n_splits, model_type):
        """
        Perform cross-validation using a general model.

        Parameters:
        - X: numpy array, the input features.
        - y: numpy array, the target variable.
        - epochs: int, the number of epochs for training.
        - n_splits: int, the number of splits for cross-validation.
        - model_type: str, the type of model to use ('mlp', 'lstm', 'simple_rnn', 'cnn').
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_no = 1
        history_list = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Your existing condition to adjust input_shape
            if model_type == 'lstm' or model_type == 'simple_rnn' or model_type == 'cnn':
                input_shape = (X_train.shape[1], 1)  # Adjusting for LSTM, SimpleRNN, CNN
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            else:
                input_shape = X_train.shape[1]  # For MLP
                
            model = None
            if model_type == 'mlp':
                model = self.build_model(input_shape)
            elif model_type == 'lstm':
                model = self.build_lstm_model((X_train.shape[1], 1))
            elif model_type == 'simple_rnn':
                model = self.build_simple_rnn_model((X_train.shape[1], 1))
            elif model_type == 'cnn':
                model = self.build_cnn_model((X_train.shape[1], 1))

            print(f'Training on fold {fold_no} with {model_type.upper()} model...')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=[early_stopping])
            
            history_list.append(history)
            fold_no += 1
        
        # After training all folds, visualize results
        return history_list

    def visualize_training_results(self, history_list, model_type):
        """
        Visualize the training results.

        Parameters:
        - history_list: list, the list of training histories for each fold.
        - model_type: str, the type of model used ('mlp', 'lstm', 'simple_rnn', 'cnn').
        """
        for i, history in enumerate(history_list):
            plt.figure(figsize=(12, 4))

            # Plot training & validation accuracy values
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title(f'Model accuracy for {model_type} - Fold {i+1}')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            # Plot training & validation loss values
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'Model loss for {model_type} - Fold {i+1}')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')

            plt.tight_layout()
            plt.show()
    def evaluate_models(self, history_list, model_types):
        """
        Evaluate the models and compare their metrics.

        Parameters:
        - history_list: list of lists, the list of training histories for each model type.
        - model_types: list of str, the types of models used.
        """
        metrics = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for i, history in enumerate(history_list):
                for hist in history:
                    plt.plot(hist.history[metric])
            plt.title(f'Comparison of model {metric}')
            plt.ylabel(metric)
            plt.xlabel('Epoch')
            plt.legend(model_types, loc='upper left')
            plt.show()