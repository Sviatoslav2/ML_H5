import numpy as np
import pandas as pd


AID1284Morered_test = pd.read_csv("./data/AID1284Morered_test.csv")
AID1284Morered_train = pd.read_csv("./data/AID1284Morered_train.csv")
AID1284Morered_test["Outcome"] = AID1284Morered_test["Outcome"].map({'Active':1, 'Inactive':0})
AID1284Morered_train["Outcome"] = AID1284Morered_train["Outcome"].map({'Active':1, 'Inactive':0})
    
AID1284red_test = pd.read_csv("./data/AID1284red_test.csv")
AID1284red_train = pd.read_csv("./data/AID1284red_train.csv")
AID1284red_test["Outcome"] = AID1284red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID1284red_train["Outcome"] = AID1284red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID1608Morered_test = pd.read_csv("./data/AID1608Morered_test.csv")
AID1608Morered_train = pd.read_csv("./data/AID1608Morered_train.csv")
AID1608Morered_test["Outcome"] = AID1608Morered_test["Outcome"].map({'Active':1, 'Inactive':0})
AID1608Morered_train["Outcome"] = AID1608Morered_train["Outcome"].map({'Active':1, 'Inactive':0})


AID1608red_test = pd.read_csv("./data/AID1608red_test.csv")
AID1608red_train = pd.read_csv("./data/AID1608red_train.csv")
AID1608red_test["Outcome"] = AID1608red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID1608red_train["Outcome"] = AID1608red_train["Outcome"].map({'Active':1, 'Inactive':0})


AID362red_test = pd.read_csv("./data/AID362red_test.csv")
AID362red_train = pd.read_csv("./data/AID362red_train.csv")
AID362red_test["Outcome"] = AID362red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID362red_train["Outcome"] = AID362red_train["Outcome"].map({'Active':1, 'Inactive':0})


AID373AID439red_test = pd.read_csv("./data/AID373AID439red_test.csv")
AID373AID439red_train = pd.read_csv("./data/AID373AID439red_train.csv")
AID373AID439red_test["Outcome"] = AID373AID439red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID373AID439red_train["Outcome"] = AID373AID439red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID373red_test = pd.read_csv("./data/AID373red_test.csv")
AID373red_train = pd.read_csv("./data/AID373red_train.csv")
AID373red_test["Outcome"] = AID373red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID373red_train["Outcome"] = AID373red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID439Morered_test = pd.read_csv("./data/AID439Morered_test.csv")
AID439Morered_train = pd.read_csv("./data/AID439Morered_train.csv")
AID439Morered_test["Outcome"] = AID439Morered_test["Outcome"].map({'Active':1, 'Inactive':0})
AID439Morered_train["Outcome"] = AID439Morered_train["Outcome"].map({'Active':1, 'Inactive':0})

AID439red_test = pd.read_csv("./data/AID439red_test.csv")
AID439red_train = pd.read_csv("./data/AID439red_train.csv")
AID439red_test["Outcome"] = AID439red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID439red_train["Outcome"] = AID439red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID456red_test = pd.read_csv("./data/AID456red_test.csv")
AID456red_train = pd.read_csv("./data/AID456red_train.csv")
AID456red_test["Outcome"] = AID456red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID456red_train["Outcome"] = AID456red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID604AID644_AllRed_test = pd.read_csv("./data/AID604AID644_AllRed_test.csv")
AID604AID644_AllRed_train = pd.read_csv("./data/AID604AID644_AllRed_train.csv")
AID604AID644_AllRed_test["Outcome"] = AID604AID644_AllRed_test["Outcome"].map({'Active':1, 'Inactive':0})
AID604AID644_AllRed_train["Outcome"] = AID604AID644_AllRed_train["Outcome"].map({'Active':1, 'Inactive':0})

AID604red_test = pd.read_csv("./data/AID604red_test.csv")
AID604red_train = pd.read_csv("./data/AID604red_train.csv")
AID604red_test["Outcome"] = AID604red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID604red_train["Outcome"] = AID604red_train["Outcome"].map({'Active':1, 'Inactive':0})


AID644Morered_test = pd.read_csv("./data/AID644Morered_test.csv")

AID644Morered_train = pd.read_csv("./data/AID644Morered_train.csv")
AID644Morered_test["Outcome"] = AID644Morered_test["Outcome"].map({'Active':1, 'Inactive':0})
AID644Morered_train["Outcome"] = AID644Morered_train["Outcome"].map({'Active':1, 'Inactive':0})

AID644red_test = pd.read_csv("./data/AID644red_test.csv")
AID644red_train = pd.read_csv("./data/AID644red_train.csv")
AID644red_test["Outcome"] = AID644red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID644red_train["Outcome"] = AID644red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID687AID721red_test = pd.read_csv("./data/AID687AID721red_test.csv")
AID687AID721red_train = pd.read_csv("./data/AID687AID721red_train.csv")
AID687AID721red_test["Outcome"] = AID687AID721red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID687AID721red_train["Outcome"] = AID687AID721red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID687red_test = pd.read_csv("./data/AID687red_test.csv")
AID687red_train = pd.read_csv("./data/AID687red_train.csv")
AID687red_test["Outcome"] = AID687red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID687red_train["Outcome"] = AID687red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID688red_test = pd.read_csv("./data/AID688red_test.csv")
AID688red_train = pd.read_csv("./data/AID688red_train.csv")
AID688red_test["Outcome"] = AID688red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID688red_train["Outcome"] = AID688red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID721morered_test = pd.read_csv("./data/AID721morered_test.csv")
AID721morered_train = pd.read_csv("./data/AID721morered_train.csv")
AID721morered_test["Outcome"] = AID721morered_test["Outcome"].map({'Active':1, 'Inactive':0})
AID721morered_train["Outcome"] = AID721morered_train["Outcome"].map({'Active':1, 'Inactive':0})

AID721red_test = pd.read_csv("./data/AID721red_test.csv")
AID721red_train = pd.read_csv("./data/AID721red_train.csv")
AID721red_test["Outcome"] = AID721red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID721red_train["Outcome"] = AID721red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID746AID1284red_test = pd.read_csv("./data/AID746AID1284red_test.csv")
AID746AID1284red_train = pd.read_csv("./data/AID746AID1284red_train.csv")
AID746AID1284red_test["Outcome"] = AID746AID1284red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID746AID1284red_train["Outcome"] = AID746AID1284red_train["Outcome"].map({'Active':1, 'Inactive':0})

AID746red_test = pd.read_csv("./data/AID746red_test.csv")
AID746red_train = pd.read_csv("./data/AID746red_train.csv")    
AID746red_test["Outcome"] = AID746red_test["Outcome"].map({'Active':1, 'Inactive':0})
AID746red_train["Outcome"] = AID746red_train["Outcome"].map({'Active':1, 'Inactive':0})

############################################################################################################# !
############################################################################################################# !
############################################################################################################# !


target1_test, target1_train = AID362red_test, AID362red_train
target2_test, target2_train = AID604red_test, AID604red_train
target3_test, target3_train = AID456red_test, AID456red_train
target4_test, target4_train = AID688red_test, AID688red_train
target5_test, target5_train = AID373red_test, AID373red_train
target6_test, target6_train = AID746AID1284red_test, AID746AID1284red_train
target7_test, target7_train = AID687red_test, AID687red_train