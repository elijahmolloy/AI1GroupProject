from keras.layers import Dense, Dropout
from keras.models import Sequential


class NeuralNetClassification:
    def __init__(self, data_loader):
        """
        This class will setup a basic neural network in an attempt
        to use binary classification with training and testing data
        as defined in the data_loader class
        :param data_loader:
        """
        self.data_loader = data_loader
        self.model = Sequential()

        # Setup and Train the model
        self.setup_model()
        self.train_model()

    def setup_model(self):
        """
        This method will use keras.layers.Dense and Dropout to setup a basic neural
        network used for classification. The output will be the 'Class' data fround

        :return:
        """
        # Add Hidden Layer One
        self.model.add(Dense(29, activation='relu', input_dim=29))
        self.model.add(Dropout(0.3))

        # Add Hidden Layer Two
        self.model.add(Dense(29, activation='relu'))
        self.model.add(Dropout(0.3))

        # Add Output Layer
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    def train_model(self):
        # Train the model
        self.model.fit(self.data_loader.train_x, self.data_loader.train_y, epochs=10)

        # Get accuracy of training data and testing data
        _, train_accuracy = self.model.evaluate(self.data_loader.train_x, self.data_loader.train_y)
        _, test_accuracy = self.model.evaluate(self.data_loader.test_x, self.data_loader.test_y)

        # Print results
        print("==============================")
        print("NEURAL NETWORK RESULTS")
        print("Test Accuracy : {0}".format(test_accuracy * 100))
