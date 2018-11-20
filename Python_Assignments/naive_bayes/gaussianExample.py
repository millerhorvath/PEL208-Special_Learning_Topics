import NaiveBayes as naive
import pandas as pd

df = pd.read_csv('gaussianExample.txt')
target_feature = 'PlayTennis'

n_bayes = naive.NaiveBayes(df, target_feature)

# n_bayes.print_probabilities()

pred_df = pd.read_csv('pred_gaussianExample.txt')

pred_list = naive.build_pred_json(data_frame=pred_df)

pred_p = n_bayes.predict_probabilities(pred_list)

pred_event = n_bayes.predict(pred_list)

for i in range(len(pred_p)):
    print(pred_df.iloc[i].to_dict())
    for k in pred_p[i]:
        print(k, pred_p[i][k])
    print('Observed Event:', df.iloc[i][target_feature])
    print('Predicted Event:', pred_event[i])
    print('')
