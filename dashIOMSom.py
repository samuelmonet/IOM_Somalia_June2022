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
usernames = ['axiom','iomsouthsudan']
passwords = ['axiomadmin','123456']

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

		comments = pd.read_csv('ML.csv', sep='\t', index_col='Code')

		select_quest = st.sidebar.selectbox('Select a question',[comments.loc[i]['Sidebar'] for i in ['well-being']+\
															['perception'+str(i) for i in range(1,6)]])
		selection = comments[comments['Sidebar']==select_quest].index.to_list()[0]

		if selection == 'well-being':
			png='shap0.png'
		else:
			png='shap{}.png'.format(selection[-1])

		commentaires = comments.loc[selection]

		title1.title('Machine learning results on predictive model trained on question:')

		title1.title(comments.loc[selection]['Question'])
		st.title('')
		st.markdown("""---""")
		st.subheader('Note:')
		st.write('A machine learning model has been run on the above mentionned question.'
				 'The objective of this is to identify, specificaly for these question, which are the the aspects of the'
				 ' project that influenced the most the responses to these question.'
				 'The figures below shows which parameters have a greater impact in the prediction '
				 'of the model than a normal random feature (following a statistic normal law)')
		st.write('')
		st.write('HOW TO READ THE GRAPH:')
		st.write('Each line of the graph represents one feature of the survey that is important to predict '
				 'the response to the question.')
		st.write('Each point on the right of the feature name represents one person of the survey. '
				 'A red point represent a high value of the specific feature and a blue point a '
				 'low value (a purple one an intermediate value).')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a '
				 'better note for the question while on the left side, this specific caracter of the person '
				 'reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and '
				 'the interpretation of the graphs is written below.')
		st.markdown("""---""")

		temp = Image.open(png)
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)

		for i in comments.loc[selection]['Coding'].split('<tab>'):
			st.caption(i)

		st.write(comments.loc[selection]['Commentaires'])

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
		text_question=pd.read_csv('questions_wc.csv')
		#st.write(text_question)
		df = data.copy()
		continues = pickle.load(open("cont_feat.p", "rb"))
		quest_list = text_question['question'].to_list()

		title1.title('Wordclouds for open questions')

		selected_quest = st.selectbox(
			'Select the question for which you would like to visualize wordclouds of answers', quest_list)

		feature=text_question[text_question['question']==selected_quest].iloc[0]['code']
		#st.write(feature)

		parenting=text_question[text_question['question']==selected_quest].iloc[0]['parent']

		#st.write(len(df))

		if parenting==parenting:
			if '=' in parenting:
				parent_feat,value=tuple(parenting.split('='))
				if value == '1':
					df=df[df[parent_feat]==1]
				else:
					df = df[df[parent_feat] == value]
			else:
				df=df[df[parenting]=='Yes']

		#st.write(len(df))
		#st.write(df)

		col_corpus = ' '.join(df[feature].apply(lambda x : '' if x in ['I do not know', 'There is no', 'None']
		else x).dropna())
		col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
		col_corpus = re.sub('\s+', ' ', col_corpus)
		col_corpus = col_corpus.lower()
		sw = st.multiselect('Select words you would like to remove from the wordclouds \n\n',
							[i[0] for i in Counter(col_corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
		if col_corpus == ' ' or col_corpus == '':
			col_corpus = 'No_response'
		else:
			col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
		wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
		wc.generate(col_corpus)
		col1, col2, col3 = st.columns([1, 4, 1])
		col2.image(wc.to_array(), use_column_width=True)

################################## A REVOIR ##################################################"

		if st.checkbox('Would you like to filter Wordcloud according to other questions'):
			dico_questions=pickle.load(open('dico_questions.p','rb'))
			feature2 = st.selectbox('Select one question to filter the wordcloud',
									[dico_questions[i] for i in dico_questions])
			filter2 = [i for i in dico_questions if dico_questions[i] == feature2][0]
			if filter2 in continues:
				a=df[filter2].astype(float)
				threshold = st.slider('Select threshold value you want to visualize',
									  min_value=float(a.min()),
									  max_value=float(a.max()),
									  value=float(a.min())
									  )
				DF=[df[df[filter2] <= threshold][feature], df[df[filter2] > threshold][feature]]
				titres=['Response under '+str(threshold),'Response over '+str(threshold)]
			else:
				DF=[df[df[filter2] == j][feature] for j in df[filter2].unique()]
				titres=['Responded : '+j for j in df[filter2].unique()]
			col1, col2 = st.columns([1, 1])
			for i in range(len(DF)):
				col_corpus = ' '.join(DF[i].dropna())
				col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
				col_corpus = re.sub('\s+', ' ', col_corpus)
				col_corpus = col_corpus.lower()
				if col_corpus == ' ' or col_corpus == '':
					col_corpus = 'No_response'
				else:
					col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(col_corpus)
				if i % 2 == 0:
					col1.subheader(titres[i])
					col1.image(wc.to_array(), use_column_width=True)
				else:
					col2.subheader(titres[i])
					col2.image(wc.to_array(), use_column_width=True)

################################################################################################

if __name__ == '__main__':
	
	hashed_passwords = stauth.Hasher(passwords).generate()
	authenticator = stauth.Authenticate(names,usernames,hashed_passwords,
    'some_cookie_name','some_signature_key',cookie_expiry_days=30)
	name, authentication_status, username = authenticator.login('Login','main')
	
	if authentication_status:
		authenticator.logout('Logout', 'main')
		main()
	elif authentication_status == False:
		st.error('Username/password is incorrect')
	elif authentication_status == None:
		st.warning('Please enter your username and password')

