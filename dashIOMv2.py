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

st.set_page_config(page_title='IOM Somalia June 2022',layout="wide")


@st.cache
def load_data():
	continues = pickle.load(open("cont_feat.p", "rb"))
	data = pd.read_csv('viz.csv', sep='\t')
	data.drop([i for i in data if 'Unnamed' in i], axis=1, inplace=True)
	correl = pd.read_csv('graphs.csv',index_col=0,sep=';')
	questions = pd.read_csv('questions.csv',index_col=0)
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
	st.sidebar.markdown(html_link("Other Dashboard",'https://axiommeltd10.shinyapps.io/IOMSomaliaBaseline_Dashboard/#section-demographics'), unsafe_allow_html=True)
	#st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")

	title1, title3 = st.columns([9,2])

	with st.sidebar:
		topic = option_menu(None, ['Machine learning results', 'Correlations', 'Wordclouds','Make your own radarplots'],
							icons=["cpu", 'bar-chart', 'cloud','heptagon'],
							menu_icon="app-indicator", default_index=0,
							)


	title3.image(img2,width=150)

	# ______________________________________ SHAP __________________________________#

	if topic == 'Machine learning results':
		
		comments = pd.read_csv('ML.csv', sep='\t', index_col='Code')

		select_quest = st.selectbox('Select a question',[comments.loc[i]['Sidebar'] for i in ['indicator'+str(i) for i in range(1,5)]])
		selection = comments[comments['Sidebar']==select_quest].index.to_list()[0]

		png='{}.png'.format(selection)
		
		
		commentaires = comments.loc[selection]

		title1.title('Machine learning results on predictive model trained on question: (select one below)')

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
		st.write('')
		st.write('You will find at the bottom of the page the metrics results which give an idea of the precision of the model used')
		
		st.markdown("""---""")

		temp = Image.open(png)
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.image(image, use_column_width = True)

		for i in comments.loc[selection]['Coding'].split('<tab>'):
			st.caption(i)

		st.write(comments.loc[selection]['Commentaires'])
		
		st.subheader('Scores of the Machine Learning models that brought these conclusions')
		col1,col2=st.columns([1,1])
		
		png2='prediction{}.png'.format(selection[-1])
		temp = Image.open(png2)
		image2 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image2.paste(temp, (0, 0), temp)
		col1.image(image2, width=700)
		col2.write(comments.loc[selection]['Results'])

	# ______________________________________ CORRELATIONS __________________________________#

	elif topic == 'Correlations':
		
		sub_topic = st.sidebar.radio('Select the topic you want to look at:',correl['categories'].unique())

		title1.title('Main correlations uncovered from the database related to {}{}'\
		.format(sub_topic,' (select a sub categorie on the sidebar)' if sub_topic == 'Districts' else ''))

		title1.write('Note: Correlation does not mean causation. This is not because 2 features are correlated that one is '
					 'the cause of the other. So conclusion have to be made with care.')
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )
		
		

		soustableau_correl = correl[correl['categories']==sub_topic]
		
		if sub_topic == 'Districts':
			sub_categ=st.sidebar.selectbox('Select what you want to look at at district level:',\
			list(soustableau_correl['categories2'].unique())[::-1])
			soustableau_correl = correl[correl['categories2']==sub_categ]
		
		if sub_topic == 'Primary Source of Income':
			st.write("It is to note that we have similar correlations at district and school level and \
			that there is also a strong correlation at these levels with primary income source. \
			Therefore, it is difficult to draw conclusions from the following correlations: Are there a consequence of\
			 the primary source of income?, of the place where people lived?, of something else correlated to both...?")

		
		st.markdown("""---""")
		k = 0
		for absc in soustableau_correl['variable_x'].unique():
			#st.write(absc)
			quest = soustableau_correl[soustableau_correl['variable_x'] == absc]
			#st.write(quest)
			
			for i in range(len(quest)):
				k+=1
				#st.write(quest.iloc[i])
				df=data.copy()
				#st.write(df.shape)
				df=select_data(quest.iloc[i],df,cat_cols)
				show_data(k,sub_topic,quest.iloc[i],df,codes)
				st.markdown("""---""")
				

	
	# ______________________________________ WORDCLOUDS __________________________________#

	elif topic == 'Wordclouds':
		title1.title('Wordcloud for question:')
		title1.title('How positive or negative is your perception regarding government role in improving access to education? Explain')
		df = data.copy()
		continues = pickle.load(open("cont_feat.p", "rb"))
		feature='expind1'
		parent='indicator1'
		col_corpus = ' '.join(df[feature].apply(lambda x : '' if x in ['I do not know', 'There is no', 'None']
		else x).dropna())
		col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
		col_corpus = re.sub('\s+', ' ', col_corpus)
		col_corpus = col_corpus.lower()
		col1, col2, col3 = st.columns([1, 3, 2])
		sw = col3.multiselect('Select words you would like to remove from the wordclouds',
							[i[0] for i in Counter(col_corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
		
		col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
		if col_corpus == ' ' or col_corpus == '':
			col_corpus = 'No_response'
			
		wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
		wc.generate(col_corpus)
		
		col2.image(wc.to_array(), use_column_width=True)
		
		border1,col1,border2, col2,border3 = st.columns([1,4,1,4,1])
		
		col1.subheader('Feeling Positive regarding government role in improving \
				access to education ({}	Respondents)'.format(len(df[df[parent]=='Positive'])))
		col2.subheader('Feeling Negative regarding government role in improving \
				access to education ({} Respondents)'.format(len(df[df[parent]=='Negative'])))
		
		for feeling in ['Positive','Negative']:# J'en suis la....
		    col_corpus = ' '.join(df[df[parent]==feeling][feature].apply(lambda x : '' if x in ['I do not know', 'There is no', 'None']
		    else x).dropna())
		    col_corpus = re.sub('[^A-Za-z ]', ' ', col_corpus)
		    col_corpus = re.sub('\s+', ' ', col_corpus)
		    col_corpus = col_corpus.lower()
		    col_corpus = ' '.join([i for i in col_corpus.split(' ') if i not in sw])
		    if col_corpus == ' ' or col_corpus == '':
		        col_corpus = 'No_response'
		        
		    wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
		    wc.generate(col_corpus)
		    if feeling=='Positive':
		    	col1.image(wc.to_array(), use_column_width=True)
		    else:
		    	col2.image(wc.to_array(), use_column_width=True)
		
	# ______________________________________ RADARS __________________________________#

	else:
		#st.write(questions)
		
		title1.title('Design your own radar plots')
		
		col1,col2= st.columns([1,1])
			
		
		cat_cols = pickle.load( open( "cat_cols.p", "rb" ) )
		subject=questions[questions['question']==col1.selectbox('Select one of the topics ',[questions.loc[i]['question'] for i in cat_cols])].index[0]
		
		questions_codes=[i for i in data if data[i].dtype=='object' and len(data[i].unique())<20]
		category=questions[questions['question']==col2.selectbox('Select a question',[questions.loc[i]['question'] for i in questions_codes])].index[0]
		
		
		
		items=data[category].unique().tolist()
		
		st.markdown("---")
		
		col1,col2= st.columns([1,1])
		
		if st.sidebar.radio('Select what you want to do:',['See all options','Compare specific groups'])=='See all options':
			
		
			for i in range(len(items)):
            
				categories = [' '.join((i.split(' ')[1:])) for i in data if subject in i[:len(subject)]]
            
				r_categ=data[data[category]==items[i]][[i for i in data if subject in i[:len(subject)]]].mean()
				r_all=data[[i for i in data if subject in i[:len(subject)]]].mean()
            
				fig2 = go.Figure()
				fig2.add_trace(go.Scatterpolar(r=r_categ, theta=categories, fill='toself', name=items[i]))
            
				fig2.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All dataset'))
            
				fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, max(r_all.max(),r_categ.max())])),showlegend=True)
				
				fig2.update_layout(title='{} : {} respondents'.format(items[i],len(data[data[category]==items[i]]))\
            				,margin={"r": 20, "t": 50, "l": 40, "b": 20})
            
				fig2.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.9, xanchor="center", x=0,\
            			 	font=dict(size=16), title=dict(font=dict(size=16),side='top'))) 
				if i%2==0:
					col1.plotly_chart(fig2,use_container_width=True)
				else:
					col2.plotly_chart(fig2,use_container_width=True)
		
		else:
			
			col1,col2=st.columns([1,3])
			
			
			col1.write('Enter the groups you want to compare (maximum 4):')
			group1=col1.multiselect('Groupe 1',['All respondents']+items)
			group2=col1.multiselect('Groupe 2',[i for i in ['All respondents']+items if i not in group1])
			group3=col1.multiselect('Groupe 3',[i for i in ['All respondents']+items if i not in group1+group2])
			group4=col1.multiselect('Groupe 4',[i for i in ['All respondents']+items if i not in group1+group2+group3])
			
			categories = [' '.join((i.split(' ')[1:])) for i in data if subject in i[:len(subject)]]
            		
			radar=go.Figure()
			r_all=data[[i for i in data if subject in i[:len(subject)]]].mean()
			maxi=r_all.max()
			if len(group1)>0:
				if 'All respondents' in group1:
					radar.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All respondents'))
				else:
					r_1=data[data[category].isin(group1)][[i for i in data if subject in i[:len(subject)]]].mean()
					radar.add_trace(go.Scatterpolar(r=r_1, theta=categories, fill='toself', name='{} <br>({}Respondents)'.\
					format(' <br>'.join(group1),len(data[data[category].isin(group1)]))))
					maxi=max(maxi,r_1.max())
			if len(group2)>0:
				if 'All respondents' in group2:
					radar.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All respondents'))
				else:
					r_2=data[data[category].isin(group2)][[i for i in data if subject in i[:len(subject)]]].mean()
					radar.add_trace(go.Scatterpolar(r=r_2, theta=categories, fill='toself', name='{} <br>({}Respondents)'.\
					format(' <br>'.join(group2),len(data[data[category].isin(group2)]))))
					maxi=max(maxi,r_2.max())
			if len(group3)>0:
				if 'All respondents' in group3:
					radar.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All respondents'))
				else:
					r_3=data[data[category].isin(group3)][[i for i in data if subject in i[:len(subject)]]].mean()
					radar.add_trace(go.Scatterpolar(r=r_3, theta=categories, fill='toself', name='{} <br>({}Respondents)'.\
					format(' <br>'.join(group3),len(data[data[category].isin(group3)]))))
					maxi=max(maxi,r_3.max())
			if len(group4)>0:
				if 'All respondents' in group4:
					radar.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All respondents'))
				else:
					r_4=data[data[category].isin(group4)][[i for i in data if subject in i[:len(subject)]]].mean()
					radar.add_trace(go.Scatterpolar(r=r_4, theta=categories, fill='toself', name='{} <br>({}Respondents)'.\
					format(' <br>'.join(group4),len(data[data[category].isin(group4)]))))
					maxi=max(maxi,r_4.max())
			
			
			radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, maxi])),showlegend=True)
			radar.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 20})
			
			
			col2.plotly_chart(radar,use_container_width=True)
	
		
		
################################################################################################

if __name__ == '__main__':
	
	main()
