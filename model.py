import pandas as pd
import numpy as np

def map_encode(df):
    df['school'] = df['school'].map({'GP': 0, 'MS': 1})
    df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    df['address'] = df['address'].map({'U': 0, 'R': 1})
    df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
    df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
    df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
    df['Fjob'] = df['Fjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
    df['reason'] = df['reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
    df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other': 2})
    df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
    df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})
    df['paid'] = df['paid'].map({'no': 0, 'yes': 1})
    df['activities'] = df['activities'].map({'no': 0, 'yes': 1})
    df['nursery'] = df['nursery'].map({'no': 0, 'yes': 1})
    df['higher'] = df['higher'].map({'no': 0, 'yes': 1})
    df['internet'] = df['internet'].map({'no': 0, 'yes': 1})
    df['romantic'] = df['romantic'].map({'no': 0, 'yes' : 1})

    return df

# Applying feature scaling on dataset
def feature_scaling(df):
    # df_scaled = df.copy()

    for i in df:
        col = df[i]
        if(np.max(col)>6):
            Max = max(col)
            Min = min(col)
            mean = np.mean(col)
            col  = (col-mean)/(Max)
            df[i] = col
        elif(np.max(col)<6):
            col = (col-np.min(col))
            col /= np.max(col)
            df[i] = col

    return df


def predict(model, data):
    try:
        preds = model.predict(data)
    except Exception as e:
        # return an error message or raise a custom exception
        return "Error occurred while predicting: " + str(e)
    return preds

# def main():
#     # Load the dataset
#     df = pd.read_csv('student-data.csv')

#     # Preprocess the data
#     X, y = preprocess_data(df)

#     # Split the dataset in train and test
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     print(X_test.shape)
#     # Train the model
#     model = train_model(X_train, y_train)
#     print(model)

#     # Test the model
#     y_pred = predict(model, X_test)

#     # Evaluate the model
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1 Score:", f1_score(y_test, y_pred))
#     print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# if __name__ == '__main__':
#     main()