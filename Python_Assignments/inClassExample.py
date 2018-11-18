from Python_Assignments.NaiveBayes import NaiveBayes
import numpy as np
import pandas as pd


def build_pred_json(data_frame):
    """

    :type data_frame: pd.DataFrame
    :param data_frame:
    :param target_feature:
    :return:
    """
    # if target_feature in data_frame.axes[1]:
    #     # _pred_dict = {'Target': target_feature, 'Event': None, 'Features': []}
    #     pass
    # else:
    #     raise Exception('Invalid target_value')

    _pred_list = []

    for idx in data_frame.axes[0]:
        _feature_dict = {}
        for feature in data_frame.axes[1]:
            if isinstance(data_frame.iloc[idx][feature], str) or not np.isnan(data_frame.iloc[idx][feature]):
                _feature_dict[feature] = data_frame.iloc[idx][feature]
        _pred_list.append(_feature_dict)

    return _pred_list


df = pd.read_csv('playTennis.txt', index_col=0)

n_bayes = NaiveBayes(df, 'PlayTennis')

# n_bayes.print_probabilities()

pred_df = pd.read_csv('pred_playTennis.txt')

pred_list = build_pred_json(data_frame=pred_df)

pred_p = n_bayes.predict_probabilities(pred_list)

pred_event = n_bayes.predict(pred_list)

for i in range(len(pred_p)):
    print(pred_df.iloc[i].to_dict())
    for k in pred_p[i]:
        print(k, pred_p[i][k])
    print('Predicted Event:', pred_event[i])
