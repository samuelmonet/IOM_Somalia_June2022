import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from fonctions import *
#import variables



		

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1,3,1])
col1.write("")
col2.title('IOM SS')
col3.write("")

st.sidebar.title('Questions Selector')


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	correl=pickle.load( open( "correlations2.p", "rb" ) )
	questions=pd.read_csv('questions.csv',index_col=0)
	questions.drop([i for i in questions if 'Unnamed' in i],axis=1,inplace=True)
	questions.columns=['parent', 'type', 'Treatment', 'Other','question']
	
	codes = pd.read_csv('codes.csv', index_col=None).dropna(how='any', subset=['color'])
	continues=pickle.load( open( "cont_feat.p", "rb" ) )
	cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
	dummy_cols=pickle.load( open( "dummy.p", "rb" ) )
	
	
	return data,correl,questions,codes,continues,cat_cols,dummy_cols


#Récup des data avec les variables nouvelles
data,correl,questions,codes,continues,cat_cols,dummy_cols=load_data()
#st.write(continues)
######################faudra surement aussi récupérer d'autres trucs sur les types de données des int_cat et int_cat_desc############################
#st.write('categorical:',cat_cols)
#st.write(correl)
#st.write(questions)
#st.write(codes)

def main():
	L=[]
	#st.write(graphs)
	q1 = st.sidebar.selectbox('Main question:', [None]+[i for i in correl if len(correl[i])>0][60:])
	if q1 != None:
		df=selectdf(data,correl,q1,cat_cols)
		
		# st.write(df)
		
		visuals=[i for i in correl[q1]]
		if len(visuals)>10:
			page=st.sidebar.selectbox('Page:', [i for i in range(len(visuals)//10,-1,-1)])
		else:
			page=0
		q2_list=visuals[page*10:page*10+10]
		
		# TRAITEMENT PARTICULIER DES DONNÉES DE CAT_COLS
		
		quests1=[i for i in df.columns if q1 in i[:len(q1)]] if q1 in cat_cols else [q1]
		
		if q1 in cat_cols:	
			st.write(df[[i for i in df.columns if q1 in i[:len(q1)]]])
			fig =px.box(df,y=[i for i in df.columns if q1 in i[:len(q1)]],points='all')
			st.plotly_chart(fig)
						
		elif q1 in ['latitude','longitude']:
			st.map(df[['latitude','longitude']],zoom=10)
		
		else:
			st.write(questions.loc[q1]['question'])
			fig=px.histogram(df, x=q1,color_discrete_sequence=['green'])
			st.plotly_chart(fig)

		#st.write(correl[q1])
#Visualisation des 6 paramètres les plus importants pour la prédiction
			
		if st.sidebar.checkbox('Do you want to generate graphs with other potential correlated questions?'):	
			
			for q2 in q2_list:
				st.write(correl[q1][q2])
				quests2=[i for i in df.columns if q2 in i] if q2 in cat_cols else [q2]
				if q2 in cat_cols:
					st.subheader(q2+': '+', '.join(quests2))			
				else:
					st.subheader(q2+': '+questions.loc[q2]['question']) #Pb avec les questions sans nom
				quest=quests1+quests2
				
				# On regarde maintenant si les deux sont des données catégorielles
					
				if q1 in cat_cols or q2 in cat_cols:
					
					if q1 in cat_cols:
						cat,autre=q1,q2
					else:
						cat,autre=q2,q1
					
					col1,col2,col3= st.columns([1,1,1])
					
					items=df[autre].unique().tolist()
					#st.write(items)
					#st.write(df[autre])
					for i in range(len(items)):
						temp = pd.DataFrame(dict(
    								r=df[df[autre]==items[i]][[i for i in df if cat in i[:len(cat)]]].mean(),
    								theta=[' '.join((i.split(' ')[1:])) for i in df if cat in i[:len(cat)]]))
						fig = px.line_polar(temp, r='r', theta='theta', line_close=True)
						fig.update_traces(fill='toself')
						fig.update_layout(title=items[i])
						if i%3==0:
							col1.plotly_chart(fig,use_container_width=True)
						elif i%3==1:
							col2.plotly_chart(fig,use_container_width=True)
						else:
							col3.plotly_chart(fig,use_container_width=True)
					
						
					
					
				elif q1 in continues:
					if q1 in ['latitude','longitude']:
						if q2 in ['latitude','longitude']:
							st.title ('Both coordinates')
						else:
							st.plotly_chart(scattermap(df,q2))
						#else:
						#	df['county']=data['county']
						#	st.plotly_chart(scattermap(df,q2,'Wau'))
						#	st.plotly_chart(scattermap(df,q2,'Bentiu'))
						#	st.plotly_chart(scattermap(df,q2,'Malakal'))
						
					else:					
						if q2 in continues:
							st.plotly_chart(scatter(q1,q2,q1,q2,df),use_container_width=True)
						else:
							st.plotly_chart(box(q1,q2,q1,q2,df),use_container_width=True)
											
				elif q2 in ['latitude','longitude']:
					st.plotly_chart(scattermap(df,q1))
				
				elif q2 in continues:
					st.plotly_chart(box(q2,q1,q2,q1,df),use_container_width=True)
					
				elif q1 in dummy_cols:
					if q2 in dummy_cols:
						df['persons']=np.ones(len(df))
						fig = px.treemap(df, path=[q1,q2],values='persons', color=q2)
						st.plotly_chart(fig, use_container_width=True)
						fig2 = px.treemap(df, path=[q2, q1], values='persons', color=q1)
						st.plotly_chart(fig2, use_container_width=True)
						
					else:
						if len(df[q1].unique())>4*len(df[q1].unique()):
							col1, col3 = st.columns([5,5])
							#col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
							#col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							
						else:
							#st.write(df)
							col1, col3 = st.columns([5,5])
							#col1.plotly_chart(count(q2,q1,q2,q1,df),use_container_width=True)
							#col3.plotly_chart(pourcent(q2,q1,q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
							col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
							col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
							
				else:
					st.write('here')
					col1, col3 = st.columns([1,1])
					#col1.plotly_chart(count(q1,q2,q1,q2,df),use_container_width=True)
					#col3.plotly_chart(pourcent(q1,q2,q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q1,q2,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q1,q2,df),use_container_width=True)
					col1.plotly_chart(count2(q2,q1,df),use_container_width=True)
					col3.plotly_chart(pourcent2(q2,q1,df),use_container_width=True)
					
			
			
			
			
		
		else:
			st.write('')
			st.write('')
			st.write('')
			col1, col2, col3 = st.columns([1,3,1])
			col2.text('Select something on the left side')
		
		
		


if __name__== '__main__':
    main()
    
