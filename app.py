import numpy as np
import pandas as pd
import streamlit as st
import pickle
from model import map_encode, feature_scaling, predict

def main():
    st.set_page_config(page_title='Student Performance Classification', page_icon=':books:')
    st.title('Student Performance Classification')
    st.write('Upload a CSV file to classify student performance.')
    file = st.file_uploader('Upload a CSV file', type=['csv'])
    if file is not None:
        df = pd.read_csv(file, delimiter=',', encoding="utf-8-sig")
        dfv = df.copy()
        model = pickle.load(open('trained_model.sav', 'rb'))
        lenc_df = map_encode(df)
        lenc_df = feature_scaling(lenc_df)

        X = lenc_df
        preds = predict(model, X)
        preds = np.where(preds == 0, 'No', 'Yes')
        st.write('Predictions:')
        st.write(preds)

        # Create a new dataframe with the predicted values
        predictions_df = pd.DataFrame({'predicted_pass': preds})

        # csv = pd.DataFrame({'Predictions': preds})
        csv = predictions_df.replace({'predicted_pass': {'no': 0, 'yes': 1}})

        # Concatenate the original dataframe with the predictions dataframe
        result_df = pd.concat([dfv, csv], axis=1)   
        rcsv_download = result_df.to_csv(index=False)
        st.download_button('Download original and predicted data as CSV', rcsv_download, 'result.csv')

        # Download the predictions as a CSV file
        csv_download = csv.to_csv(index=False)
        st.download_button('Download predictions as CSV', csv_download, 'predictions.csv')

if __name__ == '__main__':
    main()