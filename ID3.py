import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt


def traversal(root_input, count=0):
    return traversal(root_input.left_node,
                     traversal(root_input.right_node, count + 1)) if root_input is not None else count


def count_values(data_frame):
    # Gets the num of true and false values
    less_than = 0
    greater_than = 0

    try:
        # Gets the true values where >50K will be true
        greater_than = data_frame['boolean_income'].values.sum()
        # Gets the false values where <=50K will be false
        less_than = (~data_frame['boolean_income']).values.sum()
    except KeyError:
        print('Ignoring Error')
        return [less_than, greater_than]

    return [less_than, greater_than]


def calc_entropy(data_frame):
    counts = count_values(data_frame)
    less_than = counts[0]
    greater_than = counts[1]

    size_data = len(data_frame)

    true_entropy = 0.0
    if greater_than != 0:
        true_entropy = float((greater_than / size_data) * math.log(greater_than / size_data, 2))

    false_entropy = 0.0
    if less_than != 0:
        false_entropy = float((less_than / size_data) * math.log(less_than / size_data, 2))

    return (true_entropy * -1) - false_entropy


def validate_data(df):
    # check if survived column is all dead or all alive
    if len(set(df['boolean_income'].values)) == 1:
        return True

    # check if all entries in every column are the same
    for col in df:
        if col == 'boolean_income':
            continue
        if len(set(df[col].values)) != 1:
            return False

    return True


class ID3(object):

    def __init__(self, training_data, testing_data, model='dummy'):
        self.training_data = training_data
        self.testing_data = testing_data
        # Used to check if prune is needed or not
        self.model = model

    def input_stream(self, data):
        # Reads in the file and adds the headers and splits the data by the comma
        data_file = pd.read_csv(data, sep=',', names=['work_class', 'education', 'marital_status',
                                                      'occupation', 'relationship', 'race', 'gender',
                                                      'origin', 'income'])

        # Strips away the spaces in front of each value in the data set
        for i in list(data_file):
            data_file[i] = data_file[i].str.strip()

        # Adds a column of boolean_income and use that as the class value
        #   It will be set where >50K will be false and <=50K will be true
        data_file['boolean_income'] = (data_file['income'] == '<=50K')

        # Drops the income column so that entropy and info gain doesn't count it
        data_file = data_file.drop(columns='income')

        # # Creates a data file called tree_set to analyze the data after
        # if data == 'adult.data':
        #     data_file.to_csv('tree_set_train.csv', header=True)
        #
        # if data == 'adult.test':
        #     data_file.to_csv('tree_set_test.csv', header=True)

        return data_file

    def information_gain(self, data_f):

        if data_f is None:
            return None

        max_gain = -1.0
        max_column = None
        split = 0
        # Iterate through df_input, find values in the columns, and then calculate the calc_entropy on each unique value
        for col in data_f:
            len_df = len(data_f)

            # Skips 'boolean_income and 'income'
            if col == 'boolean_income':
                continue

            all_entropy_array = []
            sum_entropy = 0.0
            for value in set(data_f[col].values):
                true_subset = data_f[data_f[col] == value]
                # calculate entropy for each value
                value_entropy = calc_entropy(true_subset)

                # Calculates the summation for information gain for the set of examples
                sum_entropy = sum_entropy + ((len(true_subset) / len_df) * value_entropy)

                # Stores the values of entropy along with it's value and length of all trues and false
                counts = count_values(data_f)

                set_gain = (value_entropy, counts[0], counts[1], value)
                all_entropy_array.append(set_gain)

            # Overwrites the max_column if this gain is greater
            # Overwrites the max_gain if this gain is greater
            if (len_df - sum_entropy) > max_gain:

                max_column = col
                max_entropy = -1.0
                max_gain = (len_df - sum_entropy)

                for values in all_entropy_array:

                    # Gets the value_entropy in the set of array and compares to the max_entropy
                    if values[0] > max_entropy:
                        split = values[len(values) - 1]
                        max_entropy = values[0]

            all_entropy_array.clear()

        return Node(data_f, max_column, max_gain, split)

    def tree_build(self, df_input, depth_input, validation_data=None):

        root_assigned = self.information_gain(df_input)

        self.tree_split(root_assigned, depth_input, math.inf)

        if validation_data is not None and self.model == 'prune':
            root_assigned = self.prune(root_assigned, validation_data)

        return root_assigned

    def tree_split(self, input_root, depth_input, depth_cap):

        if input_root is None:
            return

        new_df = input_root.df

        if len(new_df.columns) == 1 or new_df.empty is True:
            input_root.find_root(new_df)
            return

        if validate_data(input_root.df) is True:
            input_root.find_root(new_df)
            return

        if depth_input >= depth_cap:
            input_root.find_root(new_df)
            return

        left_node_df = new_df[new_df[input_root.max_column] != input_root.split]
        left_node_df = left_node_df.drop(input_root.max_column, axis=1)

        if left_node_df.empty is True:
            input_root.find_root(new_df)

        right_node_df = new_df[new_df[input_root.max_column] == input_root.split]
        right_node_df = right_node_df.drop(input_root.max_column, axis=1)

        if right_node_df.empty is True:
            input_root.find_root(new_df)

        input_root.left_node = self.information_gain(left_node_df)
        self.tree_split(input_root.left_node, depth_input + 1, depth_cap)

        input_root.right_node = self.information_gain(right_node_df)
        self.tree_split(input_root.right_node, depth_input + 1, depth_cap)

    def calculate(self, node, df, mode):

        if mode == 'error':
            error = 0.0
            for index in df.iterrows():
                if node == df.loc[index]['boolean_income']:
                    continue
                else:
                    error = error + 1.0
            return float(error)
        elif mode == 'accuracy':

            num = 0
            for index, row in df.iterrows():
                num = num + self.predict(node, row)

            return num / len(df)
        else:
            print('NOT HERE')
            return 0

    def predict(self, root_node, row):

        if root_node is None:
            return 0

        if root_node.max_column == 'boolean_income':
            return 1 if root_node.split == row.loc['boolean_income'] else 0

        if root_node.split != row[root_node.max_column]:
            return self.predict(root_node.left_node, row)
        elif root_node.split == row[root_node.max_column]:
            return self.predict(root_node.right_node, row)

    def prune(self, input_root, new_df):

        if input_root is None:
            return

        left_node = None
        right_node = None

        new_df = input_root.df

        if len(new_df.columns) == 1 or new_df.empty is True:
            input_root.find_root(new_df)

        if validate_data(new_df) is True:
            input_root.find_root(new_df)

        else:
            left_node = new_df[new_df[input_root.max_column] != input_root.split]
            left_node = left_node.drop(input_root.max_column, axis=1)

            if left_node.empty is True:
                input_root.find_root(new_df)

            right_node = new_df[new_df[input_root.max_column] == input_root.split]
            right_node = right_node.drop(input_root.max_column, axis=1)

            if right_node.empty is True:
                input_root.find_root(new_df)

        input_root.right_node = self.information_gain(right_node)
        input_root.right_node = self.prune(input_root.right_node, right_node)

        input_root.left_node = self.information_gain(left_node)
        input_root.left_node = self.prune(input_root.left_node, left_node)

        return input_root


class Node(object):

    def __init__(self, data, max_column=None, information_gain=-1.0, split=0):
        self.df = data
        self.max_column = max_column
        self.information_gain = information_gain
        self.split = split
        # Initializes child nodes to none
        self.left_node = None
        self.right_node = None

    def find_root(self, data):
        self.split = 1 if count_values(data)[0] < count_values(data)[1] else 0
        self.max_column = 'boolean_income'


if __name__ == "__main__":

    # train_data = sys.argv[1]
    # test_data = sys.argv[2]
    # model = sys.argv[3]
    # training_percentage = sys.argv[4]
    #
    train_data = 'adult.data'
    test_data = 'adult.test'
    training_percentage = 50
    model = 'prune'

    data_f = ID3(train_data, test_data, model)

    if model == "vanilla":

        train_df = data_f.input_stream(train_data)
        test_df = data_f.input_stream(test_data)
        subset = train_df[0:(int(len(train_df) * int(training_percentage) / 100))]

        root = data_f.tree_build(subset, -1)

        print('Train set accuracy:\t', data_f.calculate(root, subset, 'accuracy'))
        print('Test set accuracy:\t', data_f.calculate(root, test_df, 'accuracy'))

    elif model == 'prune':

        data_f = ID3(train_data, test_data, model)
        train_df = data_f.input_stream(train_data)
        test_df = data_f.input_stream(test_data)

        subset = train_df[0:(int(len(train_df) * int(training_percentage) / 100))]

        # validation_percentage = int(sys.argv[5])
        validation_percentage = 40
        num_rows = (int(len(train_df) * int(100 - validation_percentage) / 100))
        validation_subset = train_df[num_rows:]

        depth = -1.0
        input_root = data_f.tree_build(subset, depth, validation_data=validation_subset)

        print('Train set accuracy:\t', data_f.calculate(input_root, subset, 'accuracy'))
        # print('Validation set accuracy:\t', data_f.calculate(input_root, validation_subset, 'accuracy'))
        print('Test set accuracy:\t', data_f.calculate(input_root, test_df, 'accuracy'))

    elif model == 'maxDepth':

        train_df = data_f.input_stream(train_data)
        test_df = data_f.input_stream(test_data)
        subset = train_df[:(int(len(train_df) * int(training_percentage) / 100))]

        # validation_percentage = int(sys.argv[5])
        validation_percentage = 40
        validation_subset = train_df[(int(len(train_df) * int(100 - validation_percentage) / 100)):]

        # max_depth = int(sys.argv[6])
        max_depth = 4
        # Train set accuracy:	 0.7634
        # Test set accuracy:	 0.758
        input_root = data_f.tree_build(subset, max_depth)

        print('Train set accuracy:\t', data_f.calculate(input_root, subset, 'accuracy'))
        # print('Validation set accuracy:\t', data_f.calculate(input_root, validation_subset, 'accuracy'))
        print('Test set accuracy:\t', data_f.calculate(input_root, test_df, 'accuracy'))

    else:
        pass
