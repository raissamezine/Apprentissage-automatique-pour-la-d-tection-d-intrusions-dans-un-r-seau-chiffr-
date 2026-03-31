import pandas  as pd 
import numpy as np
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
#st.set_option('deprecation.showPyplotGlobalUse',False)

def main():
    global x_train,x_test,y_train,y_test,y
    st.set_page_config(page_title="raissa &rym")

    st.title("Application de l'apprentissage automatique pour la detection d'intrusion dans un reseau")
    
    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv("df_shuffle_before_smote_een.csv")
        return data 
    df=load_data()
    df_sample =df.sample(10)
    if st.sidebar.checkbox("Afficher les données brutes :",False):
        strr="jeu de donnée : ",str(df.shape[0])
        st.subheader(strr)
        st.write(df)
    
   
if __name__=='__main__':
   
   main()