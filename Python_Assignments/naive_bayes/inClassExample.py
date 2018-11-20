import NaiveBayes as naive
import pandas as pd

df = pd.read_csv('playTennis.txt', index_col=0)

n_bayes = naive.NaiveBayes(df, 'PlayTennis')

# n_bayes.print_probabilities()

pred_df = pd.read_csv('pred_playTennis.txt')

pred_list = naive.build_pred_json(data_frame=pred_df)

pred_p = n_bayes.predict_probabilities(pred_list)

pred_event = n_bayes.predict(pred_list)

for i in range(len(pred_p)):
    print(pred_df.iloc[i].to_dict())
    for k in pred_p[i]:
        print(k, pred_p[i][k])
    print('Predicted Event:', pred_event[i])
