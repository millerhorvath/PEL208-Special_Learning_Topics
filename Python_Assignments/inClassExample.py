from Python_Assignments.NaiveBayes import NaiveBayes
import pandas as pd


df = pd.read_csv('playTennis.txt', index_col=0)

n_bayes = NaiveBayes(df, 'PlayTennis')

n_bayes.print_probabilities()
