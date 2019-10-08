import pandas as pd


class DataLoader:
    def __init__(self, train_percent=0.75):
        """
        This class will create a pandas data_frame from a file named 'creditcard.csv',
        will drop the necessary column, and will split that data_frame into a training
        and testing set.
        :param train_percent:
        """
        self.file_name = "creditcard.csv"
        self.train_x, self.train_y = [], []
        self.test_x, self.test_y = [], []

        self.data_frame = self.load_data_frame()
        self.setup_train_test_data(train_percent)

    def load_data_frame(self):
        """
        Load creditcard.csv into a data_frame, drop the Time column
        (this column is not necessary), and return the pd.data_frame
        :return: pd.data_frame of manipulated creditcard.csv
        """

        # Load .csv into a data_frame
        data_frame = pd.read_csv(self.file_name)

        # Drop 'Time' table
        data_frame = data_frame.drop(columns=["Time"], axis=1)

        return data_frame

    def setup_train_test_data(self, train_percent):
        """
        Generate a split of train vs test data from self.data_frame.
        Split both sections into input vs output arrays
        :param train_percent: split percent for train vs test
        :return:
        """

        # Split data into train vs test
        data_frame_copy = self.data_frame.copy()
        train_set = data_frame_copy.sample(frac=train_percent, random_state=0)
        test_set = data_frame_copy.drop(train_set.index)

        # Generate x and y training sets
        self.train_y = train_set.pop('Class')
        self.train_x = train_set

        # Generate x and y testing sets
        self.test_y = test_set.pop('Class')
        self.test_x = test_set
