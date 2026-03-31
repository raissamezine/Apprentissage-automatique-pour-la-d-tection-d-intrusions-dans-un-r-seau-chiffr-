import pandas as pd
import numpy as np
import streamlit as st
import os

def main():
    st.set_page_config(page_title="Raissa")

    st.title("Application de l'apprentissage automatique pour la détection d'intrusions dans un réseau")

    @st.cache_data(persist=True)
    def load_data():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        chemin = os.path.join(BASE_DIR, "df_shuffle_before_smote_een.csv")
        
        data = pd.read_csv(chemin)
        return data

    df = load_data()

    df_sample = df.sample(10)
    st.subheader("Aperçu des données")
    st.write(df_sample)

    if st.sidebar.checkbox("Afficher les données brutes :", False):
        st.subheader(f"Jeu de données : {df.shape[0]} lignes")
        st.write(df)


if __name__ == '__main__':
    main()