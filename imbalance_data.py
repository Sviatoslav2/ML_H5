import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix

def show_data_imbalance(data, target):
    Count_Normal_zero = len(data.loc[data[target] == 0])
    Count_Normal_one = len(data.loc[data[target] == 1])
    
    print("Total number 0 == ", Count_Normal_zero)
    print("Total number 1 == ", Count_Normal_one)
    
    Percentage_of_zero = Count_Normal_zero/(Count_Normal_zero + Count_Normal_one)
    print("percentage of normal transacation is ",Percentage_of_zero * 100)
    Percentage_of_one= Count_Normal_one/(Count_Normal_zero + Count_Normal_one)
    print("percentage of fraud transacation ",Percentage_of_one * 100)

    

def undersample(x, y, x_active=None, y_active=None, times=1):
    if x_active is None and y_active is None:
        x_active = x[y == 1]
        y_active = y[y == 1]
    inactive_indices= np.array(x[y == 0].index)
    count_active = len(y_active)
    inactive_indices_undersample = np.array(np.random.choice(inactive_indices,(times*count_active),replace=False))
    
    undersample_data_x = pd.concat([x.loc[inactive_indices_undersample,:], x_active])
    undersample_data_y = pd.concat([y.loc[inactive_indices_undersample], y_active])
    return undersample_data_x, undersample_data_y