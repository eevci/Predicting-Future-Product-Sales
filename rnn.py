from math import sqrt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras import *
from tensorflow.python.keras._impl.keras.layers import SimpleRNN, RNN
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

from dataset.data_utils import *


def prepareRNNModel(learning_rate, input_shape, hidden_layer, lr_decay, dropout_rate):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer, input_shape=(input_shape[1], input_shape[2]), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_layer/2), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_layer/2), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=lr_decay))
    print(model.summary())
    return model,1


def prepareRNNModel2(learning_rate, input_shape, hidden_layer, lr_decay, dropout_rate):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer, input_shape=(input_shape[1], input_shape[2]), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_layer/2), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=lr_decay))
    print(model.summary())
    return model,2


def prepareRNNModel3(learning_rate, input_shape, hidden_layer, lr_decay, dropout_rate):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer, input_shape=(input_shape[1], input_shape[2]), activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(hidden_layer, input_shape=(input_shape[1], input_shape[2]), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(hidden_layer/2), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(30))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=lr_decay))
    print(model.summary())
    return model,3


def trainWithContinousData():
    train_x_set, train_y_set, val_x_set, val_y_set, test_x_set, test_y_set = read_data('./dataset')

    print(train_x_set.shape)
    print(train_y_set.shape)
    print(val_x_set.shape)
    print(val_y_set.shape)
    print(test_x_set.shape)
    print(test_y_set.shape)
    epoch=200
    # 0.0001, input_shape, 500, 0.0001, 0.5
    def start(modelFunc, learning_rate, hidden_layer, lr_decay, dropout_rate, show_plot, verbose):
        input_shape = train_x_set.shape
        model, type = modelFunc(learning_rate, input_shape, hidden_layer, lr_decay, dropout_rate)
        history = model.fit(train_x_set, train_y_set,
                            validation_data=(val_x_set, val_y_set),
                            batch_size=100,
                            epochs=epoch,
                            verbose=verbose,
                            shuffle=False)

        y_pre = model.predict(test_x_set)
        y_pre = y_pre[:, -1]
        rmse = sqrt(mean_squared_error(test_y_set[:,-1], y_pre))
        figure_name='rnn-model%d-%d-lr%f-drop%f-lrdecay%f-epoch%d-rmse%f'%(type,hidden_layer,learning_rate,dropout_rate,lr_decay,epoch,rmse)
        model.save_weights("./models/model"+figure_name+".h5")
        if (show_plot == True):
            plotGraphs(test_y_set[:,-1], y_pre, history, figure_name+'.png')
        return 'lr=%f, hidden_layer=%d, lr_decay=%f, dropout_rate=%f ===== Val RMSE: %.3f' % (
            learning_rate, hidden_layer, lr_decay, dropout_rate, rmse)

    # lr = [0.001,0.0001]
    # hidden_layer = [64,128,256,500]
    # lr_decay = [0.001,0.0001]
    # dropout_rate = [0.4,0.5]

    lr = [0.001]
    hidden_layer = [256]
    lr_decay = [0.001]
    dropout_rate = [0.5]
    for learning_rate in lr:
        for hid_layer in hidden_layer:
            for lr_dcy in lr_decay:
                for dropout in dropout_rate:
                    for modelFunc in [prepareRNNModel2]:
                        try:
                            print(start(modelFunc, learning_rate, hid_layer, lr_dcy, dropout, True, 1))
                        except:
                            pass

def plotGraphs(test_y_set, y_pre, history, name):
    if history!= None:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','validation'], loc='upper left')
        plt.savefig('./graphs/loss-' + name+".png")
        plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(test_y_set)), test_y_set, label='Actual')
    ax.plot(np.arange(0, len(y_pre)), y_pre, label='Predicted')
    plt.title('Predicted vs Actual Sales for the Last Month')
    plt.xlabel('Month')
    ax.legend()
    plt.savefig('./graphs/'+name+".png")
    plt.show()

def predictTest():
    _, _, _, _, test_x_set, test_y_set = read_data('./dataset')

    print(test_x_set.shape)
    print(test_y_set.shape)
    model, type = prepareRNNModel2(0.001, test_x_set.shape, 256, 0.001, 0.5)
    model.load_weights('./models/rnn-best.h5')
    y_pre =model.predict(test_x_set)
    y_pre = y_pre[:, -1]
    plotGraphs(test_y_set[:, -1], y_pre, None, "")
predictTest()