import itertools
import pandas as pd
import sys


def preprocess_data(in_file):

    df = pd.read_csv(in_file)

    # print(df['goodForGroups'])
    df['goodForGroups'] = df['goodForGroups'].replace(str(1), 'gfg_1')
    df['goodForGroups'] = df['goodForGroups'].replace(str(0), 'gfg_0')

    df['open'] = df['open'].replace('TRUE', 'o_TRUE')
    df['open'] = df['open'].replace('FALSE', 'o_FALSE')

    df['delivery'] = df['delivery'].replace('TRUE', 'd_TRUE')
    df['delivery'] = df['delivery'].replace('FALSE', 'd_FALSE')

    df['waiterService'] = df['waiterService'].replace('TRUE', 'ws_TRUE')
    df['waiterService'] = df['waiterService'].replace('FALSE', 'ws_FALSE')

    df['caters'] = df['caters'].replace('TRUE', 'c_TRUE')
    df['caters'] = df['caters'].replace('FALSE', 'c_FALSE')

    return df


def gen_first_candidate(prod_map):

    return_array = []

    for key in prod_map:
        return_array.append([key])
        return_array.append(prod_map[key])

    return return_array


def input_stream(lines):
    data = []
    for line in lines:
        line = line.rstrip()
        data.append(line.split(","))

    return data


def map_product(df):
    map_prod = {}
    for data in df:
        for product in data:
            if product in map_prod:
                map_prod[product] = map_prod[product] + 1
            else:
                map_prod[product] = 1
    return map_prod


def not_in(array, whole_in):
    for index in whole_in:
        if index not in array:
            array.append(index)
    return array


class Apryori(object):

    def __init__(self, df, min_sup, min_conf, parent_array, eliminated_array):
        self.data_frame = df
        self.num_trans = len(df)
        self.minimum_support = min_sup
        self.minimum_confidence = min_conf
        self.parent_array = parent_array
        self.eliminated_items = eliminated_array

    def gen_frequent_items(self, candidate_array):
        freq_items = []
        for index in range(len(candidate_array)):

            if index % 2 != 0:
                support = float(candidate_array[index]) / self.num_trans

                if (support * 100) < self.minimum_support:
                    self.eliminated_items.append(candidate_array[index - 1])
                else:
                    freq_items.append(candidate_array[index - 1])
                    freq_items.append(candidate_array[index])

        for index in freq_items:
            self.parent_array.append(index)

        f_item_len = len(freq_items)

        if f_item_len != 0 and f_item_len != 2:
            self.gen_candidates(freq_items)
        else:
            return self.parent_array

    def gen_candidates(self, frequent_items):

        only_elements = []
        for index in range(len(frequent_items)):
            if index % 2 == 0:
                only_elements.append(frequent_items[index])

        post_combos = []
        for item in only_elements:

            temp = []
            j = only_elements.index(item)

            for index in range(j + 1, len(only_elements)):

                temp = not_in(temp, item)
                temp = not_in(temp, only_elements[index])
                post_combos.append(temp)

                temp = []

        sorted_combos = []
        for index in post_combos:
            sorted_combos.append(sorted(index))

        unique_combos = []
        unique_combos = not_in(unique_combos, sorted_combos)

        post_combos = unique_combos

        candidate_array = []
        for item in post_combos:
            count = 0

            for transaction in self.data_frame:
                if set(item).issubset(set(transaction)):
                    count = count + 1

            if count != 0:
                candidate_array.append(item)
                candidate_array.append(count)

        self.gen_frequent_items(candidate_array)

    def find_association(self):

        association_array = []
        for item in self.parent_array:

            if isinstance(item, list) and len(item) > 1:

                length = len(item) - 1
                while length > 0:

                    temp = []
                    for RHS in list(itertools.combinations(item, length)):
                        temp.append(list(set(item) - set(RHS)))
                        temp.append(list(RHS))
                        association_array.append(temp)
                        temp = []

                    length = length - 1

        return association_array

    def final_output(self):

        output_set = []

        rules = self.find_association()
        rules.sort()

        for rule in rules:
            rule.sort()

            x_count = 0
            xy_count = 0

            for transaction in self.data_frame:
                # Gets the X value
                if set(rule[0]).issubset(set(transaction)):
                    x_count = x_count + 1

                # Gets the X and Y value
                if set(rule[0] + rule[1]).issubset(set(transaction)):
                    xy_count = xy_count + 1

            x_support = round((float(x_count) / self.num_trans) * 100, 2)
            xy_support = round((float(xy_count) / self.num_trans) * 100)
            confidence = (xy_support / x_support) * 100

            if confidence < self.minimum_confidence:
                continue

            # supportOfXAppendString = "Support Of X: " + str(x_support)
            output_set.append("Support Of X: " + str(x_support))

            # supportOfXandYAppendString = "Support of X & Y: " + str(xy_support)
            output_set.append("Support of X & Y: " + str(xy_support))

            # confidenceAppendString = "Confidence: " + str(round(confidence))
            output_set.append("Confidence: " + str(round(confidence)))

            output_set.append(rule)

        return output_set


if __name__ == '__main__':

    # file_input = sys[1]
    # min_support = float(sys[2])
    # min_confidence = float(sys[3])

    file_input = '../yelp5.csv'
    min_support = 25
    min_confidence = 75

    # df = preprocess_data(file_input)
    # df = df.to_csv(encoding='utf-8', index=False)

    with open(file_input) as fp:
        lines = fp.readlines()

    # Takes away the header
    lines = lines[1:]

    # Pre-processes the data
    dataSet = input_stream(lines)

    parent_frequent = []
    eliminated_items = []

    # Initalizes the object
    apriori = Apryori(dataSet, min_support, min_confidence, parent_frequent, eliminated_items)

    first_candidate = gen_first_candidate(map_product(dataSet))

    frequent_items = apriori.gen_frequent_items(first_candidate)

    AprioriOutput = apriori.final_output()

    counter = 1
    for i in AprioriOutput:
        if counter == 4:
            print(str(i[0]) + "------>" + str(i[1]))
            counter = 0
        else:
            print(i, end='  ')
        counter = counter + 1
