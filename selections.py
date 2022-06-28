import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

questions=pd.read_csv('questions.csv',index_col=0)
data = pd.read_excel('graphs.xlsx',index_col=0)

#st.write(' '.join(data.columns))
st.write(questions)

variable_x = st.text_input('variable_x')
variable_y = st.text_input('variable_y')
graphtype=st.selectbox('Type de graph',['radar','map','bar','treemap','violin'])

categories = st.text_input('Categorie')

if len(variable_x)>1 and len(variable_y)>1:
	title = st.text_input('Title of graph',questions.loc[variable_x]['question'] +' VS '+questions.loc[variable_y]['question'])
Description = st.text_input('Description')

xtitle,ytitle,legendtitle='','',''

if graphtype == 'bar':
	legendtitle = st.text_input('Title of legend',questions.loc[variable_y]['question'])

if graphtype == 'violin':
	ytitle = st.text_input('ytitle',questions.loc[variable_y]['question'])

if graphtype in ['violin','bar']:
	xtitle = st.text_input('xtitle',questions.loc[variable_x]['question'])
		
if st.button('Validate'):
	data.loc[len(data)]=[variable_x,variable_y,categories,Description,xtitle,ytitle,legendtitle,title,graphtype]
	data.to_excel('graphs.xlsx')
	
	st.write(len(data))
