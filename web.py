import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

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
  st.header('Using Streamlit')

  PassengerId=st.text_input('PassengerId')
  Age=st.slider('Age',0,70)

  sex_encoder=LabelEncoder()
  embarked_encoder=LabelEncoder()

  Sex=st.radio('Gender',['Male','Female','Prefer not to say'])
  Embarked=st.selectbox('Embarked',['C','Q','S'])

  Sex=sex_encoder.fit_transform([Sex])[0]
  Embarked=embarked_encoder.fit_transform([Embarked])[0]

  SibSp=st.number_input('SibSp',0,5)
  Pclass=st.number_input('Pclass',0,5)
  Parch=st.number_input('Parch',0,5)
  Fare=st.text_input('Fare','0')
  
  st.sidebar.title('Visualization')
  columns=['line','scatter','bar']
  for columns in st.sidebar.multiselect('Choose Plots',columns):
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

  with st.spinner('loading...'):
    if st.button('Survival Test Result'):
      survive=survival_prediction([Age,SibSp,Pclass,Parch,PassengerId,Sex,Fare,Embarked])
      time.sleep(3)
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

st.sidebar.title('Settings')
st.sidebar.title('Contact us')  
