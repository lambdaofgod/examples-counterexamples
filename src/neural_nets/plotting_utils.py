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

        loss = model.losses

        if add_smoothing:
            smoothed_loss = self._smooth_ma(loss)
            plt.plot(smoothed_loss)
            print('Final loss, smoothed: {:.4f}'.format(smoothed_loss[-1]))
        else:
            plt.plot(loss)

        plt.show()

        y_pred = model.predict(self.X_test)
        print('accuracy: {:.4f}'.format(model.score(self.X_test, self.y_test)))
        print(classification_report(self.y_test, y_pred))


    @staticmethod
    def _smooth_ma(series, window_width=10):
        """
        Smooths input with moving average of window_width
        """
        pad_width = (window_width // 2, window_width // 2)

        # pad the array so that numpy moving average makes sense for the whole range
        padded_series = np.pad(series, pad_width=pad_width, mode='edge')

        return np.convolve(
            padded_series,
            v=np.ones(window_width) / window_width,
            mode='valid')