from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassification:
    def __init__(self, data_loader):
        """
        This class will setup a logistic regression model in an attempt
        to use binary classification with training and testing data
        as defined in the data_loader class
        :param data_loader:
        """
        self.data_loader = data_loader

        # Setup and Train model
        self.setup_and_train_model()

    def setup_and_train_model(self):
        """
        This method will setup and train sklearn's logistic regression using the
        training data from data_loader and will score the model using the testing data
        from data_loader. The method will print the score
        :return:
        """
        # Training and Testing data from data_loader
        train_x, train_y = self.data_loader.train_x, self.data_loader.train_y
        test_x, test_y = self.data_loader.test_x, self.data_loader.test_y

        # Setup and Fit model
        model = LogisticRegression()
        model.fit(train_x, train_y)

        # Test Model
        test_accuracy = model.score(test_x, test_y)

        # Print score
        print("==============================")
        print("LOGISTIC REGRESSION RESULTS")
        print("Test Accuracy : {0}".format(test_accuracy * 100))
