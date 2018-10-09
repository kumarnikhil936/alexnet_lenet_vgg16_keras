import numpy as np

from lfw_fuel import lfw
from models import  alexnet
from clean import clean
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from keras.models import load_model


# Load the data, shuffled and split between train and test sets
(X_train_orig, y_train_orig), (X_test_orig, y_test_orig) = lfw.load_data("funneled")

# Preprocess the images 
(X_train, y_train), (X_test, y_test) = clean(X_train_orig, y_train_orig, X_test_orig, y_test_orig)

tb = TensorBoard(log_dir='./alexnet_logs',
                  write_graph=True,
                  histogram_freq=1,
                  write_images=True,
                  embeddings_freq=0)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


history = LossHistory()

batch_size = 100
num_epochs = 50 

print("\n")

# modelB = alexnet()
# modelB.fit(X_train, y_train,
#            batch_size=batch_size,
#            epochs=num_epochs,
#            verbose=1,
#            validation_data=(X_test, y_test),
#            callbacks=[tb, history])
# print('Saving the model now')
# modelB.save('alexnet.h5')
# print('Done saving the model')
# del modelB;
modelB = load_model('alexnet.h5')
modelB.summary()
score = modelB.evaluate(X_test, y_test, verbose=1)
print("-" * 40)
print("Alexnet Model (%d epochs):" % num_epochs)
print('Test accuracy: {0:%}'.format(score[1]))
print("\n")
y_predicted = modelB.predict(X_test)
print("Zeros: %d" % (np.sum(y_test == 0)))
print("Ones: %d" % (np.sum(y_test == 1)))
print("\n")
print("Simple Model : Predicted")
print("Zeros: %d" % (np.sum(y_predicted < 0.4)))
print("Not Sure: %d" % (np.sum(np.logical_and(y_predicted > 0.4, y_predicted < 0.6))))
print("Ones: %d" % (np.sum(y_predicted > 0.6)))
print("\n")


#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#yy = history.losses
#
#xx = range(len(yy))
#ax.plot(xx, yy)
#ax.set_xlabel('Batch')
#ax.set_ylabel('Loss')
#
