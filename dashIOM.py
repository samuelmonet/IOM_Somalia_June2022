import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pickle
import re
from collections import Counter
from PIL import Image
from streamlit_option_menu import option_menu
from dashboard_fonctions import *
import streamlit_authenticator as stauth

st.set_page_config(layout="wide")


@st.cache
def load_data():
	continues = pickle.load(open("cont_feat.p", "rb"))
	data = pd.read_csv('viz.csv', sep='\t')
	data.drop([i for i in data if 'Unnamed' in i], axis=1, inplace=True)
	correl = pd.read_excel('graphs.xlsx',index_col=0)
	questions = pd.read_csv('questions.csv')
	questions.drop([i for i in questions if 'Unnamed' in i],axis=1,inplace=True)
	questions.columns=['parent', 'type', 'Treatment', 'Other','question']
	codes = pd.read_csv('codes.csv', index_col=None).dropna(how='any', subset=['color'])
	return data, correl, questions, codes



data, correl, questions, codes = load_data()

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoIOM.png")

names = ['Axiom','IOM']
usernames = ['axiom','iomsomalia']
passwords = ['axiomadmin','schools']

#st.write(questions)

def main():
	#st.write(codes)
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")

	title1, title3 = st.columns([9,2])

	with st.sidebar:
		topic = option_menu(None, ['Machine learning results', 'Correlations', 'Wordclouds'],
							icons=["cpu", 'bar-chart', 'cloud'],
							menu_icon="app-indicator", default_index=0,
							)


	title3.image(img2)

	# ______________________________________ SHAP __________________________________#

	if topic == 'Machine learning results':

		st.title('Not ready')


	# ______________________________________ CORRELATIONS __________________________________#

	elif topic == 'Correlations':
		
		sub_topic = st.sidebar.radio('Select the topic you want to look at:',correl['categories'].unique())

		title1.title('Main correlations uncovered from the database related to '+sub_topic)

		title1.write('Note: Correlation does not mean causation. This is not because 2 features are correlated that one is '
					 'the cause of the other. So conclusion have to be made with care.')
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )
		
		

		soustableau_correl = correl[correl['categories']==sub_topic]
		
		if sub_topic=='Districts':
				sub_categ=st.selectbox('Select what you want to look at at district level:',soustableau_correl['categories2'].unique())
				soustableau_correl = correl[correl['categories2']==sub_categ]
			

		st.markdown("""---""")
		k = 0
		
		

		st.markdown("""---""")
		k = 0
		for absc in soustableau_correl['variable_x'].unique():
			#st.write(absc)
			quest = soustableau_correl[soustableau_correl['variable_x'] == absc]
			#st.write(quest)
			
			for i in range(len(quest)):
				#st.write(quest.iloc[i])
				df=data.copy()
				#st.write(df.shape)
				df=select_data(quest.iloc[i],df,cat_cols)
				#st.write(df)
				show_data(quest.iloc[i],df,codes)
				st.markdown("""---""")
				

	
	# ______________________________________ WORDCLOUDS __________________________________#

	elif topic == 'Wordclouds':
		st.title('Not ready')
		
################################################################################################

if __name__ == '__main__':
	
	hashed_passwords = stauth.Hasher(passwords).generate()
	authenticator = stauth.Authenticate(names,usernames,hashed_passwords)
	name, authentication_status, username = authenticator.login('Login','main')
	
	if authentication_status:
		authenticator.logout('Logout', 'main')
		main()
	elif authentication_status == False:
		st.error('Username/password is incorrect')
	elif authentication_status == None:
		st.warning('Please enter your username and password')

