import os
from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional


class RNN(Network):
    def train_model(self, unroll):
        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1, patience=1)

        model = Sequential()
        model.add(Bidirectional(LSTM(self.hidden_nodes[0], unroll=unroll, return_sequences=True), input_shape=(self.x_train.shape[1:])))
        for i in range(1, len(self.hidden_nodes) - 1):
            model.add(LSTM(self.hidden_nodes[i], return_sequences=True,  unroll=unroll))

        if self.n_layers > 1:
            model.add(Bidirectional(LSTM(self.hidden_nodes[-1], unroll=unroll)))

        model.add(Dense(self.y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        model.fit(self.x_train, self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  epochs=self.epochs, shuffle=False,
                  batch_size=2048, callbacks=[es, lr_reduce])
        return model, self.rnn_params_to_folder(unroll)

    def rnn_params_to_folder(self, unroll) -> str:
        folder_name = f"LSTM_{str(self.n_layers)}_{'-'.join([str(i) for i in self.hidden_nodes])}" \
                      f"_{self.feature_name}_context_length_{self.context_length}_unroll_{unroll}"
        if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
            os.mkdir(folder_name)
        return folder_name
