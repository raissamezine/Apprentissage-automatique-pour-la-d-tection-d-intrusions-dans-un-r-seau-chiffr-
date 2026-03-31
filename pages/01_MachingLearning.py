
import pandas  as pd 
import numpy as np
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score,confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import metrics

import seaborn as sns
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@st.cache_data(persist=True)
def load_data():
        chemin = os.path.join(BASE_DIR, "df_shuffle_before_smote_een.csv")
        df=pd.read_csv(chemin)
        df=df.dropna()
        df=df.drop_duplicates()
        df=df.iloc[:-1,:]
        df=df.sample(frac=1).reset_index(drop=True)
        return df 
df=load_data() 
wrappers=["RF","DT","LGBM"]
filters=["MIC","CHI2","pearson"]
def afficherResult(prediction):
     precision =precision_score(y_test, prediction, average='macro').round(2)
     accuracy=accuracy_score(y_test,prediction).round(3)
     recall=recall_score(y_test,prediction).round(3)
     F1_scr=f1_score(y_test,prediction).round(3)
     # Afficher le résultat
     st.write("Précision :",precision)
     st.write("Accuracy :",accuracy)
     st.write("Recall :",recall)
     st.write("F1_score :",F1_scr)
     ## afficher les graphiques de performances 
     plot_perf(graphe_perf)
def plot_perf(graphes):
        if "matrice de confusion" in graphes:
            st.subheader( "Matrice de confusion")
            
            confusion_matrix = metrics.confusion_matrix(y_test, prediction)
            matrix_df = pd.DataFrame(confusion_matrix)

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
            ax.set_title('Confusion Matrix')
            ax.set_xlabel("Predicted label", fontsize=15)
            labels = y.unique()
            labels = [''] + list(labels)
            tick_locs = np.arange(len(labels))
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(labels, rotation=0)
            ax.set_ylabel("True Label", fontsize=15)
            ax.set_yticklabels(list(labels), rotation=0)

            # Display the confusion matrix plot
            st.pyplot(fig)

        if "courb ROC" in graphes:
            st.subheader("courb ROC")
            y_scores = model.predict_proba(x_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Plot diagonal line for random classifier
            ax.set_title('ROC Curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()

            # Display the ROC curve plot
            st.pyplot(fig)
            
        if "Courbe precision_Recall" in graphes:
            st.subheader("Courbe precision_Recall")
           
            st.pyplot()
def extractfeatures(type,méthode,fullname):
     df_temp=df
     chemin = os.path.join(BASE_DIR, "FVs", type, fullname)
     data = pd.read_csv(chemin)
     
     if méthode=="RF" or méthode=="MIC": num=1
     if méthode=="DT" or méthode=="CHI2": num=2
     if méthode=="LGBM" or méthode =="pearson": num=3
     
     feat=[]
     for i in range (len(data)):
      
        
        if data.iloc[i][num]==True: 
            
             feat.append(data.iloc[i][0])
     print("les features selectionnées :",feat)
     features= df_temp.columns

     for f in features:
      if f != "Classe":
         if f not in feat :
              del  df_temp[f]
     return df_temp
def combinationoffeatures(type,méthode,fullname):
     df_temp=df
     chemin = os.path.join(BASE_DIR, "FVs", type, fullname)
     data = pd.read_csv(chemin) 
     relevants=[]
     for row in data.itertuples(index=False):
           if row.count(True)>1:
                  relevants.append(row[0]) 
     print("les features selectionnées :",relevants)
     features= df_temp.columns

     for f in features:
      if f != "Classe":
         if f not in relevants :
              del  df_temp[f]
     return df_temp  
def split_data(dff):
       
       x=dff.drop('Classe',axis=1)
       y=dff["Classe"] 
       x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1,random_state=42)
       return  x_train,x_test,y_train,y_test,y
    
def returnmodel(classifier):
     if classifier=="Random Forest":
                   model =RandomForestClassifier(n_estimators=35) 
     if classifier=="Decision tree":
                   model = DecisionTreeClassifier(max_depth =30)
     if (classifier=="KNN"):
                    model = KNeighborsClassifier(n_neighbors=5)
     if (classifier=="XGboost"):
                   model = XGBClassifier(n_estimators=100,eta=0.3)
     if (classifier=="AdaBoost"):
                   model = AdaBoostClassifier(n_estimators=350, random_state=42, learning_rate=0.38)
     if (classifier=="GradientBoost"):
                    model = GradientBoostingClassifier(n_estimators=40,learning_rate=0.1)
     if (classifier=="LGBM"):
                    model = LGBMClassifier(n_estimators=300,learning_rate=0.1,max_depth=30)
     if (classifier=="Logistic Regression"):
                    model=LogisticRegression(max_iter=1500,solver='newton-cholesky', penalty='l2', C=10)
     return model
def meilleure(chemin,num_methode):
          df_temp=df
          
          chemin = os.path.join(BASE_DIR, chemin)
          data = pd.read_csv(chemin)
          feat=[]
          for i in range (len(data)):
      
        
           if data.iloc[i][num_methode]==True: 
            
             feat.append(data.iloc[i][0])
          print("les features selectionnées :",feat)
          features= df_temp.columns

          for f in features:
            if f != "Classe":
             if f not in feat :
              del  df_temp[f]
          return df_temp,feat
classifier=st.sidebar.selectbox("Classificateur",("Random Forest","Decision tree","KNN","XGboost","AdaBoost","GradientBoost"
                                                  ,"LGBM","Logistic Regression"))
graphe_perf=st.sidebar.multiselect("choisir un graphe de performance du modèle ML",
                                           ("matrice de confusion","courb ROC"))

choix=st.sidebar.radio("",['Méthodes de selection de caractéristiques','Meilleur vecteur de caractéristiques'])

if choix=="Méthodes de selection de caractéristiques":
    méthode=st.sidebar.selectbox("Méthode",("RF","DT","LGBM","MIC","CHI2","pearson","RDL","MCP",'PCA'))
    nb_car=st.sidebar.number_input("nombre caractéristique",1,43,step=1)
    if st.sidebar.button("Execution",key="classify"):
   
         if méthode in wrappers:
              fullname="wrapper_trees"+str(nb_car)+".csv"
              d=extractfeatures("Wrappers",méthode,fullname)
              d.to_csv("resultt.csv",index=False)
              x_train,x_test,y_train,y_test,y=split_data(d)
              model=returnmodel(classifier)
              rslt="Resultat du "+classifier+" en utilisant la méthode de selection: "+méthode
              st.subheader(rslt)
              # entrainement de l'algorithme
              model.fit(x_train,y_train)
                # la prédiction 
              prediction=model.predict(x_test)
              afficherResult(prediction)

         if méthode in filters:
            fullname="filters"+str(nb_car)+".csv"
            d=extractfeatures("filters",méthode,fullname)
            d.to_csv("resultt.csv",index=False)
            x_train,x_test,y_train,y_test,y=split_data(d)
            model=returnmodel(classifier)
            rslt="Resultat du "+classifier+" en utilisant la méthode de selection: "+méthode
            st.subheader(rslt)
              # entrainement de l'algorithme
            model.fit(x_train,y_train)
                # la prédiction 
            prediction=model.predict(x_test)
            afficherResult(prediction)

         if méthode == "RDL":
               fullname="wrapper_trees"+str(nb_car)+".csv"
               d=combinationoffeatures("Wrappers",méthode,fullname)
               x_train,x_test,y_train,y_test,y=split_data(d)
               model=returnmodel(classifier)
               rslt="Resultat du "+classifier+" en utilisant la méthode de selection: "+méthode
               st.subheader(rslt)
              # entrainement de l'algorithme
               model.fit(x_train,y_train)
                # la prédiction 
               prediction=model.predict(x_test)
               afficherResult(prediction)
         if méthode == "MCP":
                fullname="filters"+str(nb_car)+".csv"
                d=combinationoffeatures("filters",méthode,fullname)
                x_train,x_test,y_train,y_test,y=split_data(d)
                model=returnmodel(classifier)
                rslt="Resultat du "+classifier+" en utilisant la méthode de selection: "+méthode
                st.subheader(rslt)
               # entrainement de l'algorithme
                model.fit(x_train,y_train)
                    # la prédiction 
                prediction=model.predict(x_test)
                afficherResult(prediction)
         if méthode =="PCA":
                x_train,x_test,y_train,y_test,y=split_data(df)
                numf=nb_car
                pca = PCA(n_components=numf)
                X_train_pca = pca.fit_transform(x_train)
                X_test_pca = pca.transform(x_test)
                model=returnmodel(classifier)
                model.fit( X_train_pca,y_train)
                # la prédiction 
                prediction=model.predict(X_test_pca)
                afficherResult(prediction)

if choix=="Meilleur vecteur de caractéristiques":
    col1,col2=st.columns([1,2])
    col1.markdown("# Essai :")
    col1.markdown("nouvelle instance  :")
    data_file=col2.file_uploader("import CSV",type=["csv"])
    if data_file is not None:
      col2.success("file uploaded successfully ")
      with st.expander("Details"):
        
        data=pd.read_csv(data_file )
        st.dataframe(data)

    if st.sidebar.button("Execution",key="classify"):
       if classifier=="Random Forest":
           d,features= meilleure(os.path.join("FVs","filters","filters20.csv"),1)
       if classifier=="Decision tree":
          d,features= meilleure(os.path.join("FVs","filters","filters20.csv"),1)
       if classifier=="KNN":
           d,features= meilleure(os.path.join("FVs","filters","filters20.csv"),1)
       if classifier=="XGboost":
           d,features= meilleure(os.path.join("FVs","filters","filters20.csv"),1)

       if classifier=="LGBM":
           d,features= meilleure("FVs/filters/filters30.csv",1)
       if classifier=="Logistic Regression":
           print("logistiqueeeeeee")
           d,features= meilleure("FVs/filters/filters40.csv",2)
       if classifier=="AdaBoost":
             d=df
             features=df.columns
       if classifier=="GradientBoost":
             d=df
             features=df.columns
       x_train,x_test,y_train,y_test,y=split_data(d)
       model=returnmodel(classifier)
       rslt="Resultat du "+classifier
       st.subheader(rslt) 
       model.fit(x_train,y_train)          
       prediction=model.predict(x_test)
       if data_file is not None:
        df_temp=data 
        featuress= df_temp.columns

        for f in featuress:
            if f != "Classe":
              if f not in features :
                 del  df_temp[f]
        real_classe=df_temp.loc[0][-1]
        #vec=df_temp.loc[0][:-1].tolist().reshape(1, -1)
        vec = np.array(df_temp.loc[0][:-1].tolist()).reshape(1,len(features))
        pre=model.predict(vec)
        if pre==0 : 
            res="cette est connexion est bénigne 0"
            st.subheader(res)
            if real_classe==pre:
                  st.success("True")
            else :st.error("False")
        if pre==1 : 
            res="cette est connexion est malicieuse 1"
            st.subheader(res)
            if real_classe==pre:
                  st.success("True")
            else :st.error("False")
       afficherResult(prediction)
       
            

