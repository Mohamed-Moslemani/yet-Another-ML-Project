import matplotlib.pyplot as plt
import seaborn as sns


class Visualizations:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, column):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()
    
    def plot_scatter(self, column1, column2):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data[column1], y=self.data[column2])
        plt.title(f'Scatter Plot of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_pairplot(self):
        sns.pairplot(self.data)
        plt.suptitle('Pair Plot', y=1.02)
        plt.show()

    def plot_bar(self, column):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.data[column])
        plt.title(f'Bar Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()
    