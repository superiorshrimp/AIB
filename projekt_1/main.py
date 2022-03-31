#link do zbioru danych: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

from printData import print_sample_train_letter
from printData import print_test_letters_and_pred
from createData import getData
from modelBuild import build_model

 
data_path = "D:\Data\A_ZHandwrittenData.csv" # "C:\\Users\\monik\\OneDrive\\Pulpit\\AIB2\\AIB\\data.csv"

x_train_data, x_test_data, categorical_train_data, categorical_test_data, shuffled = getData("D:\Data\A_ZHandwrittenData.csv")

print_sample_train_letter(shuffled)

model = build_model(x_train_data, x_test_data, categorical_train_data, categorical_test_data)

print_test_letters_and_pred(model, x_test_data, categorical_test_data)
