import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

if __name__ == '__main__':

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    # Create a model.
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Number of training samples.
    sample_count = 12

    # Total epochs to train.
    epochs = 7

    # Number of warmup epochs.
    warmup_epoch = 5

    # Training batch size, set small value here for demonstration purpose.
    batch_size = 4

    # Generate dummy data.
    data = np.random.random((sample_count, 100))
    labels = np.random.randint(10, size=(sample_count, 1))

    # Convert labels to categorical one-hot encoding.
    one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

    # Compute the number of warmup batches.
    warmup_batches = warmup_epoch * sample_count / batch_size

    # Create the Learning rate scheduler.
    warm_up_lr = WarmUpLearningRateScheduler(warmup_batches, init_lr=0.001)

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, one_hot_labels, epochs=epochs, batch_size=batch_size,
            verbose=0, callbacks=[warm_up_lr])