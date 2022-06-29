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



#img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoIOM.png")

#st.write(questions)

def main():
	#st.write(codes)
	axiom_html = get_img_with_href('logoAxiom.png', 'https://axiom.co.ke')

	st.sidebar.markdown(axiom_html, unsafe_allow_html=True)
	st.sidebar.markdown(html_link("Other Dashboard",'http://axiomdashboard.com/shap.html'), unsafe_allow_html=True)
	#st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")

	title1, title3 = st.columns([9,2])

	with st.sidebar:
		topic = option_menu(None, ['Machine learning results', 'Correlations', 'Wordclouds','Make your own radarplots'],
							icons=["cpu", 'bar-chart', 'cloud','heptagon'],
							menu_icon="app-indicator", default_index=0,
							)


	title3.image(img2)

	# ______________________________________ SHAP __________________________________#

	if topic == 'Machine learning results':
		
		
		
		comments = pd.read_csv('ML.csv', sep='\t', index_col='Code')

		select_quest = st.selectbox('Select a question',[comments.loc[i]['Sidebar'] for i in ['indicator'+str(i) for i in range(1,5)]])
		selection = comments[comments['Sidebar']==select_quest].index.to_list()[0]

		png='{}.png'.format(selection)

		commentaires = comments.loc[selection]

		title1.title('Machine learning results on predictive model trained on question:')

		st.write('Attention!!!!!!!, comment do not yet match the grphs, however graphs are correct') 

		st.title(comments.loc[selection]['Question'])
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
		st.markdown(html_link("More details on SHAP can be found here",'http://axiomdashboard.com/shap.html'), unsafe_allow_html=True)
		
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
				sub_categ=st.sidebar.selectbox('Select what you want to look at at district level:',\
				list(soustableau_correl['categories2'].unique())[::-1])
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
				if sub_topic=='Districts':
					second=st.selectbox('Look at a lower level for '+str(i),[None,'village','school','clan','subdistrict'])
				else:
					second=None
				#st.write(quest.iloc[i])
				df=data.copy()
				#st.write(df.shape)
				df=select_data(quest.iloc[i],df,cat_cols)
				st.write(second)
				show_data(quest.iloc[i],df,codes,second=second)
				
				
				
				st.markdown("""---""")
				

	
	# ______________________________________ WORDCLOUDS __________________________________#

	elif topic == 'Wordclouds':
		st.title('Not ready')
		
		
	# ______________________________________ WORDCLOUDS __________________________________#

	else:
		title1.title('Design your own radar plots')
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )
		subject=st.selectbox('Select one of the topics ',cat_cols)
		
		category=st.selectbox('Select a category',[i for i in data if data[i].dtype=='object' and len(data[i].unique())<20])
		
		if st.radio('Select what you want to do:',['See all options','Select specific options'])=='See all options':
			col1,col2= st.columns([1,1])
			items=data[category].unique().tolist()
		
			for i in range(len(items)):
            
				categories = [' '.join((i.split(' ')[1:])) for i in data if subject in i[:len(subject)]]
            
				r_categ=data[data[category]==items[i]][[i for i in data if subject in i[:len(subject)]]].mean()
				r_all=data[[i for i in data if subject in i[:len(subject)]]].mean()
            
				fig2 = go.Figure()
				fig2.add_trace(go.Scatterpolar(r=r_categ, theta=categories, fill='toself', name=items[i]))
            
				fig2.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All dataset'))
            
				fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, max(r_all.max(),r_categ.max())])),showlegend=True)

            
            
            
            
				if i%2==0:
					col1.plotly_chart(fig2,use_container_width=True)
				else:
					col2.plotly_chart(fig2,use_container_width=True)
		
		
	
	
		
		
################################################################################################

if __name__ == '__main__':
	
	main()
