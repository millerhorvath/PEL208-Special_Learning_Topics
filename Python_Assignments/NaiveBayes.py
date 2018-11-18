import numpy as np
import pandas as pd
from pandas import DataFrame


class NaiveBayes:
    def __init__(self, data_frame=None, target_feature=None):
        """

        :type data_frame: pd.DataFrame
        :param data_frame:
        :type target_feature: str
        :param target_feature:
        """
        self.indP = None  # Individual categorical events probability
        self.condP = None  # Conditional probability of events related with the target feature
        self.df = data_frame  # Original DataFrame
        self.target = target_feature  # Target feature
        self.fitted = False  # Flag to control whether the model was trained

        if self.df is not None and self.target is not None:
            self.compute_probabilities()
        else:
            print("WARNING: Can't fit Naive Bayes model without the DataFrame and the target_value.")

    def fit(self, data_frame, target_feature):
        """

        :type data_frame: pd.DataFrame
        :param data_frame:
        :type target_feature: str
        :param target_feature:
        """
        self.fitted = False
        self.df = data_frame
        self.target = target_feature

        self.compute_probabilities()

    def print_probabilities(self):
        print('##### Individual Probabilities')
        for k in self.indP.keys():
            print(k, self.indP[k])
        print('')
        print('##### Conditional Probabilities for', self.target)
        for k in self.condP.keys():
            print(k, self.condP[k])

    def compute_probabilities(self):
        # Initialize probability dicts
        self.indP = {}
        self.condP = {}

        # Cast object columns into categorical features
        for k in self.df.keys():
            if self.df[k].dtype == np.object:
                self.df[k] = self.df[k].astype('category')

        n = len(self.df)  # Number of observation in data

        # Compute individual events probability
        for feature in self.df.axes[1]:
            if self.df[feature].dtype == 'category':
                for event in self.df[feature].cat.categories:
                    self.indP[feature, event] = len(self.df[self.df[feature] == event]) / n

        # Group original DataFrame by the target feature (must be categorical, for now)
        grouped = self.df.groupby(self.target)

        # Compute conditional probabilities
        for target, data in grouped:
            n = len(data)  # Number of observations for each target feature event

            for feature in data.keys():
                # Only computes for categorical features
                if feature != self.target:
                    # Iterate for all feature events
                    for event in data[feature].cat.categories:
                        # cond_prob_dict['Feature'].append(feature)
                        # cond_prob_dict['Value'].append(event)
                        # cond_prob_dict['Target'].append(target)
                        # p = len(data[data[feature] == event]) / n
                        # cond_prob_dict['P'].append(p)
                        self.condP[feature, event, target] = len(data[data[feature] == event]) / n

        self.fitted = True
