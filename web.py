import streamlit as st
import joblib
import pandas as pd
import numpy as np
df=pd.read_csv('/content/titanic.csv')

model=joblib.load('/content/Rand.joblib')

from sklearn.preprocessing import LabelEncoder
a=LabelEncoder()

def survival_prediction(input_data):
  
  a=np.asarray(input_data)
  b=a.reshape(1,-1)

  prediction=model.predict(b)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person is not Suvived'
  else:
    return 'The person is Survived'

def main():
  st.title('Survival Prediction')
  Age=st.slider('Age',0,70)
  SibSp=st.slider('SibSp',0,5)
  Pclass=st.slider('Pclass',0,5)
  Parch=st.slider('Parch',0,5)
  PassengerId=st.text_input('PassengerId')
  sex_encoder=LabelEncoder()
  embarked_encoder=LabelEncoder()
  Sex=st.selectbox('Sex',['Male','Female'])
  Sex=sex_encoder.fit_transform([Sex])[0]
  Fare=st.text_input('Fare','0')
  Embarked=st.selectbox('Embarked',['C','Q','S'])
  Embarked=embarked_encoder.fit_transform([Embarked])[0]

  
  columns=['line','scatter','bar']
  for columns in st.multiselect('choose a plot',columns):
    if columns=='line':
      st.title('Lineplot')
      st.line_chart(df['Fare'])    
    elif columns=='scatter':
      st.title('Scatterplot')
      st.scatter_chart(df['Fare'])
    elif columns=='bar':
      st.title('Barplot')
      st.bar_chart(df['Fare'])
  survive=''

  if st.button('Survival Test Result'):
    survive=survival_prediction([Age,SibSp,Pclass,Parch,PassengerId,Sex,Fare,Embarked])

  st.success(survive)

if __name__ == '__main__':
  main()

st.markdown("""
<style>
.main {
  background-color: purple;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title('Profile')
st.sidebar.title('Settings')
st.sidebar.title('Help')  