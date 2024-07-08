import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

rad=st.sidebar.radio("Navigation Menu",["Home","Diabete","Graphique"])

#Home Page
if rad=="Home":
    st.title(" Application de prediction du diabete ")
    st.image("Medical Prediction Home Page.jpg")
   

#Diabetes Prediction
df2=pd.read_csv("Diabetes Predictions.csv")
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
model2=RandomForestClassifier()
model2.fit(x2_train,y2_train)

#Diabetes Page
if rad=="Diabete":
    st.header("Savoir si vous êtes touché par le diabète")
    st.write("Toutes les valeurs doivent être dans la plage mentionnée")
    glucose=st.number_input("Entrez votre niveau de glucose (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Entrez votre niveau d'insuline dans le corps (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Entrez votre indice de masse corporelle/valeur IMC (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Entrez votre âge (20-80)",min_value=20,max_value=80,step=1)
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]

    if st.button("Predire"):
        if prediction2==1:
            st.warning("Vous etes affecté par le diabète")
        elif prediction2==0:
            st.success("Vous en sécurité")

     
                                        
if rad=="Grahique":
    type=st.selectbox("Graphique",["Diabete"])

    if type=="Diabete":
        fig=px.scatter(df2,x="Glucose",y="Outcome")
        st.plotly_chart(fig)
    
