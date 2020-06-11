import pandas as pd
import joblib

def new_features(X):
    X['AgeBuilt'] = X['YrSold'] - X['YearBuilt'] #age of house since it was built
    X['AgeGarageBlt'] = X['YrSold'] - X['GarageYrBlt'] #age of garage since it was built
    X['AgeRemodeled'] = X['YrSold'] - X['YearRemodAdd'] #age of house since it was remodeled

    drop_var1 = ['YrSold','YearBuilt','GarageYrBlt','YearRemodAdd']
    X.drop(drop_var1, axis=1, inplace = True)
      
    return X

def num_to_cat(X):
    X['MSSubClass'] = X['MSSubClass'].astype(str)
    X['MoSold'] = X['MoSold'].astype(str)
    return X

def nan_to_none(X): 
    
    features_with_none_level = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                                'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish',
                                'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
    
    for col_name in features_with_none_level:    
        X[col_name] = X[col_name].fillna('None')

    return X

def handle_missing_values(X): 

    for fill_0 in ['MasVnrArea', 'GarageCars', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 
                    'BsmtFinSF2', 'TotalBsmtSF', 'GarageArea', 'BsmtUnfSF']:
        X[fill_0] = X[fill_0].fillna(0)

    median_value = joblib.load("/content/drive/My Drive/House/median_value.pkl") #loads a dictionary of feature-median pair

    #replace NA in numerical features with median of the corresponding feature in training set
    for num_col in list(X.select_dtypes(exclude='object')):
        X[num_col] = X[num_col].fillna(median_value[num_col])

    mode_value = joblib.load("/content/drive/My Drive/House/mode_value.pkl") #loads a dictionary of feature-mode pair

    #replace NA in categorical features with mode of the corresponding feature in training set
    for categ_col in list(X.select_dtypes(include='object')):
        X[categ_col] = X[categ_col].fillna(mode_value[categ_col]) 
        
    return X

def ordinal_encoding(X):
    X['Functional'] = X['Functional'].map({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0})
    X['Alley'] = X['Alley'].map({'Pave':2,'Grvl':1,'None':0})
    X['Street'] = X['Street'].map({'Pave':1,'Grvl':0})
    X['LotShape'] = X['LotShape'].map({'Reg':3, 'IR1':2,'IR2':1,'IR3':0})
    X['Utilities'] = X['Utilities'].map({'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':1}) 
    X['LandSlope'] = X['LandSlope'].map({'Gtl':2,'Mod':1,'Sev':0})
    X['ExterQual'] = X['ExterQual'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    X['ExterCond'] = X['ExterCond'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    X['BsmtQual'] = X['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
    X['BsmtCond'] = X['BsmtCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
    X['BsmtExposure'] = X['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0})
    X['BsmtFinType1'] = X['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
    X['BsmtFinType2'] = X['BsmtFinType2'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'None':0})
    X['HeatingQC'] = X['HeatingQC'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    X['CentralAir'] = X['CentralAir'].map({'N':0,'Y':1})
    X['KitchenQual'] = X['KitchenQual'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0})
    X['FireplaceQu'] = X['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
    X['GarageFinish'] = X['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'None':0})
    X['GarageQual'] = X['GarageQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
    X['GarageCond'] = X['GarageCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0})
    X['PavedDrive'] = X['PavedDrive'].map({'Y':2,'P':1,'N':0})
    X['PoolQC'] = X['PoolQC'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1,'None':0})
    
    return X

def feature_scaling(X):
  numerical_columns = list(X.select_dtypes(exclude='object'))
  fs = joblib.load("/content/drive/My Drive/House/robust_scaler.pkl")
  X[numerical_columns] = fs.transform(X[numerical_columns]) 

  return X

def dummy_coding(X):
    return pd.get_dummies(X)

def fix_columns(X):
    # Find columns that exist in train set, but not in test set
    train_features = joblib.load('/content/drive/My Drive/House/train_features.pkl')
    missing_cols = set(train_features) - set(X.columns) 

    # Add the missing column in test set and set it equal to 0
    for col in missing_cols: 
        X[col] = 0

    # Select only the columns (features) that exist in train set, with the correct order
    X = X[train_features] 

    return X

def full_transform(X):
    X = new_features (X) # create new features
    X = num_to_cat (X) # convert some numerical features to categorical
    X = nan_to_none (X) # convert NA to None for some features
    X = handle_missing_values (X) # handle missing values in all features
    X = ordinal_encoding (X) # ordinal encoding of some categorical features
    X = feature_scaling (X) # robust scale numerical features
    X = dummy_coding (X) # dummy coding of some categorical features
    X = fix_columns (X) # ensures that new dataset contains only the features found in training set

    return X