import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

class ScoringProgram:
    def __init__(self, real_values, evaluated_values):
        self.real_values = real_values
        self.evaluated_values = evaluated_values
        self.accuracy = None
        self.conf_matrix = None

    def calculate_accuracy(self):
        self.accuracy = accuracy_score(self.real_values, self.evaluated_values)
        self.conf_matrix = confusion_matrix(self.real_values, self.evaluated_values)

    def display_results(self):
        print("Accuracy:", self.accuracy)
        print("Confusion Matrix:\n", self.conf_matrix)

    def plot_confusion_matrix(self):
        plt.imshow(self.conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = np.unique(self.real_values)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(self.conf_matrix[i, j]), ha='center', va='center', color='white' if self.conf_matrix[i, j] > len(self.real_values)//2 else 'black')

        plt.show()

        hgsfgdsf