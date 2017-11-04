import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class Plotter:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def plot_loss_and_accuracy(self, model, add_smoothing=True):
        plt.plot(model.losses)

        plt.title('Training loss (and smoothed) across iterations')
        plt.xlabel('No steps')
        plt.ylabel('Loss')
        print('Final loss: {:.4f}'.format(model.losses[-1]))

        if add_smoothing:
            ma_width = 5

            # pad the array so that numpy moving average makes sense for the whole range
            pad_width = (ma_width // 2, ma_width // 2)
            padded_losses = np.pad(model.losses, pad_width=pad_width, mode='edge')

            smoothed_sgd_loss = np.convolve(
                padded_losses,
                v=np.ones(ma_width) / ma_width,
                mode='valid')
            plt.plot(smoothed_sgd_loss)
            print('Final loss, smoothed: {:.4f}'.format(smoothed_sgd_loss[-1]))
        plt.show()

        y_pred = model.predict(self.X_test)
        print('accuracy: {:.4f}'.format(model.score(self.X_test, self.y_test)))
        print(classification_report(self.y_test, y_pred))
