# Apprentissage-automatique-pour-la-d-tection-d-intrusions-dans-un-r-seau-chiffr-
Ce projet a pour objectif de détecter des intrusions dans un trafic réseau chiffré en utilisant des techniques de Machine Learning et Deep Learning.

À partir de données réseau brutes (paquets PCAP), nous utilisons Zeek pour extraire des logs structurés, qui sont ensuite exploités pour construire un système de détection d’intrusions.

---

##  Pipeline du projet


###  Étapes :

1.  Capture du trafic réseau (fichiers PCAP)  
2.  Analyse avec Zeek → génération de logs :
   - `conn.log` (connexions)
   - `ssl.log` (trafic chiffré)
   - `weird.log` (comportements anormaux)
3.  Prétraitement et nettoyage des données  
4.  Extraction et sélection de caractéristiques  
5.  Entraînement de modèles de Machine Learning  
6.  Évaluation des performances  

---

##  Méthodes utilisées

###  Sélection de caractéristiques
- Méthodes filtres : Chi2, Pearson, MIC  
- Méthodes wrappers : Random Forest, Decision Tree, LGBM  
- Combinaisons : RDL, MCP  
- Réduction de dimension : PCA  

###  Modèles de classification
- Random Forest  
- Decision Tree  
- KNN  
- XGBoost  
- LightGBM  
- AdaBoost  
- Gradient Boosting  
- Régression Logistique  

---

##  Évaluation

Les performances des modèles sont évaluées avec :
- Accuracy  
- Précision  
- Recall  
- F1-score  
- Matrice de confusion  
- Courbe ROC  

---

##  Application Streamlit

Une interface interactive a été développée avec Streamlit permettant de :

- Choisir le modèle de classification  
- Sélectionner la méthode de sélection de caractéristiques  
- Visualiser les performances  
- Tester de nouvelles instances (fichier CSV)  

---


---

##  Exécution du projet

streamlit run Home.py
##  Dataset

Le fichier `df_shuffle_before_smote_een.csv` correspond au **dataset final** utilisé dans ce projet.

Il est obtenu après :
- le prétraitement des logs réseau extraits avec Zeek  
- le nettoyage des données 
- le mélange (shuffle) des données  

Ce dataset constitue la base pour l’entraînement et l’évaluation des modèles de Machine Learning.
