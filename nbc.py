#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import sys


class NaiveBayes:

    # Used for calling the file, getting the binary values, and dropping none column
    def input_stream(self, df):
        # Imports the data and limits the Columns to only:
        # ambience, parking, dietaryRestrictions, recommendedFor
        # and replaces every null/empty value to 'None'

        df = pd.read_csv(df).replace({'[]': 'None'}).fillna('None')

        # ambienceDummy =  # 9 Columns
        df = pd.concat([df, NaiveBayes.regex(self, 'ambience', df)], axis=1)
        # print(df)

        # parkingDummy =  # 5 Columns
        df = pd.concat([df, NaiveBayes.regex(self, 'parking', df)], axis=1)
        # print(df)

        # dietaryRestrictionsDummy =   # 7 Columns
        df = pd.concat([df, NaiveBayes.regex(self, 'dietaryRestrictions', df)], axis=1)
        # print(df)

        # recommendedForDummy =   # 6 Columns
        df = pd.concat([df, NaiveBayes.regex(self, 'recommendedFor', df)], axis=1)
        # print(df)

        df = df.drop(columns=['ambience', 'parking', 'dietaryRestrictions', 'recommendedFor'])
        # print(df) # 43 columns

        return df

    # Used for calling and changing up the column names to match the overall schema of names
    def regex(self, columnName, df):
        df[columnName] = df[columnName].str.replace('\]|\[', '')
        df[columnName] = df[columnName].str.replace(',\s*', ',')
        df[columnName] = df[columnName].str.replace('\'', '')
        result = df[columnName].str.get_dummies(',').drop(columns="None")
        return result

    # classifier
    def conditional_probabilities(self, training_df, test_df):
        storeValue = NaiveBayes.build_map(self, training_df)
        priors = NaiveBayes.class_priors(self, training_df)

        ZOLoss = 0.0
        SQLoss = 0.0

        for row in range(len(test_df)):

            tp = priors[0]
            fp = priors[1]

            for col in test_df.columns:
                if col == 'outdoorSeating':
                    continue
                else:
                    try:
                        val = test_df.at[row, col]
                        tp *= storeValue[(True, col, val)]
                        fp *= storeValue[(False, col, val)]
                    except KeyError:
                        tp = tp
                        fp = fp

            tp_norm = float(tp) / float(tp + fp)
            fp_norm = float(fp) / float(fp + tp)

            if tp_norm < fp_norm:
                guess = False
            else:
                guess = True

            if test_df.at[row, 'outdoorSeating'] != guess:
                ZOLoss += 1

            if test_df.at[row, 'outdoorSeating']:
                SQLoss += (1.0 - tp_norm) * (1 - tp_norm)
            else:
                SQLoss += (1.0 - fp_norm) * (1 - fp_norm)

        ZOLoss = ZOLoss / len(test_df)
        SQLoss = SQLoss / len(test_df)

        print("ZERO-ONE LOSS=%f" % ZOLoss)
        print("SQUARED LOSS=%f" % SQLoss)

    # Builds the map
    def build_map(self, df):
        storeValue = {}

        for columns in df.columns:
            for values in df[columns].unique():
                #          (OS     column    val)
                # Finds the probability of the true values
                storeValue[(True, columns, values)] = NaiveBayes.calc_prob(self, df, True, columns, values)
                # Finds the probability of the false values
                storeValue[(False, columns, values)] = NaiveBayes.calc_prob(self, df, False, columns, values)

        return storeValue

    # Calculates the probability
    def calc_prob(self, train_df, outdoorSeatingValue, columns, val):

        os_df = train_df.groupby([columns, 'outdoorSeating']).size().div(len(train_df)).to_dict()

        priors = NaiveBayes.class_priors(self, train_df)

        try:
            if (val, outdoorSeatingValue) in os_df.keys():
                return float(os_df[val, outdoorSeatingValue])
            else:
                if outdoorSeatingValue:
                    return 1.0 / (len(train_df[columns].unique()) + priors[1])
                else:
                    return 1.0 / (len(train_df[columns].unique()) + priors[0])
        except:
            print()

    # Calculates class priors in [ true, false]
    def class_priors(self, df):
        # get number of 1s in outdoor seating
        # get number of 0s in outdoor seating
        # divide both numbers by number of rows
        outdoorDF = df['outdoorSeating'].value_counts()
        count_class_prior = df['outdoorSeating'].count()

        # x/A = class_prior_true
        class_prior_true = float(outdoorDF[1]) / float(count_class_prior)
        # y/B = class_prior_false
        class_prior_false = float(outdoorDF[0]) / float(count_class_prior)
        # print(class_prior_true, class_prior_false)
        return [class_prior_true, class_prior_false]


if __name__ == '__main__':
    nbc = NaiveBayes()

    train_data = sys.argv[1]
    test_data = sys.argv[2]

    export = pd.DataFrame.from_dict(nbc.build_map(nbc.input_stream(train_data)), orient="index")
    export.to_csv('export.csv')

    cp = nbc.class_priors(nbc.input_stream(train_data))
    nbc.conditional_probabilities(nbc.input_stream(train_data), nbc.input_stream(test_data))
