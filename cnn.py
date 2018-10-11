from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
import numpy as np
import os
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D


class CNN(Network):

    def train_model(self, kernel_sizes, strides, dense_size=128):
        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1, patience=1)
        self.x_train = np.expand_dims(self.x_train, 3)
        self.x_val = np.expand_dims(self.x_val, 3)
        self.x_test = np.expand_dims(self.x_test, 3)
        model = Sequential()
        model.add(Conv2D(self.hidden_nodes[0], kernel_size=kernel_sizes[0], strides=strides[0],
                         activation='relu', padding='same',
                         input_shape=(self.x_val.shape[1:])))
        for i in range(1, len(self.hidden_nodes)):
            model.add(Conv2D(self.hidden_nodes[i], padding='same', kernel_size=kernel_sizes[i], strides=strides[i]))

        model.add(Flatten())
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dense(self.y_val.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(self.x_train, self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  epochs=self.epochs,
                  batch_size=2048, callbacks=[es, lr_reduce])
        return model, self.cnn_params_to_folder(kernel_sizes, strides, dense_size)

    def cnn_params_to_folder(self, kernel_sizes, strides, dense_size) -> str:
        folder_name = f"CNN_{str(self.n_layers)}_{'-'.join([str(i) for i in self.hidden_nodes])}" \
                      f"_{self.feature_name}_context_length_{self.context_length}" \
                      f"_strides_{'_'.join([str(i) for i in strides])}" \
                      f"_kernels_{'-'.join([str(i) for i in kernel_sizes])}_denselayer_{dense_size}"
        if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
            os.mkdir(folder_name)
        return folder_name
