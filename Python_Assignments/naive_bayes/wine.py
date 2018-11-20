import NaiveBayes as naive
import pandas as pd
import os

df = pd.read_csv(os.path.join('..', 'wine.txt'))
target_feature = 'class'

n_bayes = naive.NaiveBayes(df, target_feature)

# n_bayes.print_probabilities()

# pred_df = pd.read_csv('pred_gaussianExample.txt')
pred_df = df.iloc[:, 0:-1].copy()

pred_list = naive.build_pred_json(data_frame=pred_df)

pred_p = n_bayes.predict_probabilities(pred_list)

pred_event = n_bayes.predict(pred_list)

target_list = n_bayes.df[n_bayes.target].cat.categories
confusion_matrix = {}

for t in target_list:
    confusion_matrix[t] = {}
    for t2 in target_list:
        confusion_matrix[t][t2] = 0

for i in range(len(pred_p)):
    print(pred_df.iloc[i].to_dict())
    for k in pred_p[i]:
        print(k, pred_p[i][k])
    obs = df.iloc[i][target_feature]
    pred = pred_event[i]
    print('Observed Event:', obs)
    print('Predicted Event:', pred)
    print('')
    confusion_matrix[obs][pred] += 1


for t in target_list:
    print('{}'.format(t), end=' ')
print('')

for t in target_list:
    for t2 in target_list:
        print('\t{}'.format(confusion_matrix[t][t2]), end='')
    print('')
