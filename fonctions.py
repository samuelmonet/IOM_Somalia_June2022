import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pydeck as pdk


#import variables

codes = pd.read_csv('codes.csv', index_col=None).dropna(how='any', subset=['color'])

#Maps
def scattermap(df,feat):
	fig = px.scatter_mapbox(df, lat="latitude", lon="longitude",  color=feat,
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=14)
	fig.update_layout(mapbox_style="open-street-map")
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	return fig

#Fonction de graph
def sunb(q1,i,main_question,second_question,dfm):
	dfm['ones']=np.ones(len(dfm))
	fig = px.sunburst(dfm.fillna(''), path=[q1,i], values='ones')
	fig.update_layout(title_text=main_question + ' et ' +second_question,font=dict(size=20))
	fig.update_layout(title_text=q1+ ' et ' +i,font=dict(size=20))
	return fig

def count(q1,q2,main_question,second_question,dfm):
	agg=dfm[[q2,q1]].groupby(by=[q1,q2]).aggregate({q1:'count'}).unstack()
	x=[i for i in agg.index]
	fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
	for i in range(len(agg.columns)-1):
    		fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
	fig.update_layout(barmode='relative', \
                  xaxis={'title':main_question},\
                  yaxis={'title':'Persons'}, legend_title_text=None)
	return fig

def count2(abscisse,ordonnée,dataf):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T.round(2)*100
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':'<b>'+abscisse+'<b>','title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.1,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text='test')
    
    return fig


def pourcent(q1,q2,main_question,second_question,dfm):
	agg=dfm[[q2,q1]].groupby(by=[q1,q2]).aggregate({q1:'count'}).unstack()
	agg=agg.T/agg.T.sum()
	agg=agg.T*100
	x=[i for i in agg.index]
	fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
	for i in range(len(agg.columns)-1):
    		fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
	fig.update_layout(barmode='relative', \
                  xaxis={'title':main_question},\
                  yaxis={'title':'Pourcentages'}, legend_title_text=None)
	return fig


def pourcent2(abscisse,ordonnée,dataf):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['code'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} people",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':'<b>'+abscisse+'<b>','title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.1,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text='test')
    
    return fig


def box(cont,cat,cont_question,noncont_question,dfm):
	fig = px.box(dfm, x=cat, y=cont,points='all')
	fig.update_traces(marker_color='green')
	fig.update_layout(barmode='relative', \
                  xaxis={'title':noncont_question},\
                 yaxis_title=cont_question)
	return fig


def scatter(q1,q2,main_question,second_question,dfm):
	fig = px.scatter(dfm, x=q1, y=q2)
	fig.update_layout(xaxis={'title':main_question},yaxis_title=second_question)

	return fig


def selectdf(data,correl,q1,cat_cols):
	q2_list=[i for i in correl[q1]]+[q1]
	features=[]
	df=data.copy()
	
	for feat in q2_list:
		if feat in cat_cols:
			features+=[k for k in data.columns if feat in k]
		else:
			features.append(feat)
	if 'latitude' in features and 'longitude' not in features:
		features.append('longitude')
	if 'longitude' in features and 'latitude' not in features:
		features.append('latitude')
	return df[features]



def selectdf2(q1,q2,dfm,cat_cols):
	if q1 in cat_cols and q2 in cat_cols:
		df2=pd.DataFrame(columns=[q1,q2])
		quests1=[i for i in dfm.columns if q1 in i]
		catq1=[' '.join(i.split(' ')[1:]) for i in quests1]
		
		#st.write(quests1)
		quests2=[i for i in dfm.columns if q2 in i]
		catq2=[' '.join(i.split(' ')[1:]) for i in quests2]
		#st.write(quests2)
		#st.write(dfm[quests1+quests2])
		for i in range(len(quests1)):
			for j in range(len(quests2)):       
				ds=dfm[[quests1[i],quests2[j]]].copy()
				ds=ds[ds[quests1[i]].isin(['Yes',1])]
				ds=ds[ds[quests2[j]].isin(['Yes',1])]      
				ds[quests1[i]]=ds[quests1[i]].apply(lambda x: catq1[i])
				ds[quests2[j]]=ds[quests2[j]].apply(lambda x: catq2[j])
				ds.columns=[q1,q2]
				df2=df2.append(ds)	
	
	else:
		if q1 in cat_cols:
			cat,autre=q1,q2
		else:
			cat,autre=q2,q1
		if autre != 'longitude':
			df2 = pd.DataFrame(columns=[cat, autre])


			catcols=[j for j in dfm.columns if cat in j]
			#st.write(catcols)
			cats=[' '.join(i.split(' ')[1:]) for i in catcols]
			#st.write(cats)
		
			for i in range(len(catcols)):
				ds=dfm[[catcols[i],autre]].copy()
				ds=ds[ds[catcols[i]].isin(['Yes',1])]
				ds[catcols[i]]=ds[catcols[i]].apply(lambda x: cats[i])
				ds.columns=[cat,autre]
				df2=df2.append(ds)
			#st.write(df2)
		else:
			df2 = pd.DataFrame(columns=[cat, 'latitude','longitude'])

			catcols = [j for j in dfm.columns if cat in j]
			# st.write(catcols)
			cats = [' '.join(i.split(' ')[1:]) for i in catcols]
			# st.write(cats)

			for i in range(len(catcols)):
				ds = dfm[[catcols[i], 'latitude','longitude']].copy()
				ds = ds[ds[catcols[i]].isin(['Yes', 1])]
				ds[catcols[i]] = ds[catcols[i]].apply(lambda x: cats[i])
				ds.columns = [cat, 'latitude','longitude']
				df2 = df2.append(ds)
	# st.write(df2)

	return df2
