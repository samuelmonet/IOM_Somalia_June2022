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
from joypy import joyplot

x, y = np.ogrid[100:500, :600]
mask = ((x - 300) / 2) ** 2 + ((y - 300) / 3) ** 2 > 100 ** 2
mask = 255 * mask.astype(int)

def sankey_graph(datas, L, height=600,width=1600):
	""" sankey graph de data pour les catégories dans L dans l'ordre et  de hauter et longueur définie éventuellement"""
	nodes_colors = ["blue", "green", "grey", 'yellow', "coral", 'darkviolet', 'saddlebrown', 'darkblue', 'brown']
	link_colors = ["lightblue", "limegreen", "lightgrey", "lightyellow", "lightcoral", 'plum', 'sandybrown', 'lightsteelblue', 'rosybrown']
	labels = []
	source = []
	target = []
	for cat in L:
		lab = datas[cat].unique().tolist()
		lab.sort()
		labels += lab
	for i in range(len(datas[L[0]].unique())):  # j'itère sur mes premieres sources
		source+=[i for k in range(len(datas[L[1]].unique()))]  # j'envois sur ma catégorie 2
		index=len(datas[L[0]].unique())
		target+=[k for k in range(index,len(datas[L[1]].unique())+index)]
		for n in range(1, len(L)-1):
			source += [index+k for k in range(len(datas[L[n]].unique())) for j in range(len(datas[L[n+1]].unique()))]
			index += len(datas[L[n]].unique())
			target += [index+k for j in range(len(datas[L[n]].unique())) for k in range(len(datas[L[n+1]].unique()))]
	iteration = int(len(source)/len(datas[L[0]].unique()))
	value_prov = [(int(i//iteration), source[i], target[i]) for i in range(len(source))]
	value = []
	k = 0
	position = []
	for i in L:
		k += len(datas[i].unique())
		position.append(k)
	for triplet in value_prov:
		k = 0
		while triplet[1] >= position[k]:
			k += 1
		df = datas[datas[L[0]] == labels[triplet[0]]].copy()
		df = df[df[L[k]] == labels[triplet[1]]]
		value.append(len(df[df[L[k+1]] == labels[triplet[2]]]))
	color_nodes=nodes_colors[:len(datas[L[0]].unique())]+["black" for i in range(len(labels)-len(datas[L[0]].unique()))]
	color_links=[]
	for i in range(len(datas[L[0]].unique())):
		color_links += [link_colors[i] for couleur in range(iteration)]
	fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=30, line=dict(color="black", width=1),
											  label=[i.upper() for i in labels], color=color_nodes),
									link=dict(source=source, target=target, value=value, color=color_links))])
	return fig


def count2(abscisse, ordonnee, dataf, codes, legendtitle='', xaxis=''):
    dataf[ordonnee] = dataf[ordonnee].apply(lambda x: str(x))
    agg = dataf[[abscisse, ordonnee]].groupby(by=[abscisse, ordonnee]).aggregate({abscisse: 'count'}).unstack().fillna(
        0)
    agg2 = agg.T / agg.T.sum()
    agg2 = agg2.T * 100
    agg2 = agg2.astype(int)
    if abscisse == 'Village_clean':
        agg = agg.reindex(['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood",
                           'Al-Samoud'])
        agg2 = agg2.reindex(
            ['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood",
             'Al-Samoud'])

    x = agg.index

    if ordonnee.split(' ')[0] in codes['list name'].values:
        # st.write('on est là')
        colors_code = codes[codes['list name'] == ordonnee.split(' ')[0]].sort_values(['code']).copy()
        labels = colors_code['label'].tolist()
        colors = colors_code['color'].tolist()
        fig = go.Figure()
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnee].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse, str(labels[i]))], name=str(labels[i]),
                                     marker_color=colors[i].lower(), customdata=agg2[(abscisse, str(labels[i]))],
                                     textposition="inside",
                                     texttemplate="%{customdata} %", textfont_color="black"))
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:, 0], name=agg.columns.tolist()[0][1], marker_color='green',
                               customdata=agg2.iloc[:, 0], textposition="inside",
                               texttemplate="%{customdata} %", textfont_color="black"))
        for i in range(len(agg.columns) - 1):
            fig.add_trace(
                go.Bar(x=x, y=agg.iloc[:, i + 1], name=agg.columns.tolist()[i + 1][1], customdata=agg2.iloc[:, i + 1],
                       textposition="inside", texttemplate="%{customdata} %", textfont_color="black"))
    fig.update_layout(barmode='relative', xaxis={'title': xaxis, 'title_font': {'size': 18}},
                      yaxis={'title': 'Persons', 'title_font': {'size': 18}}
                      )
    fig.update_layout(legend_title=legendtitle, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center",
                                                            x=0.5, font=dict(size=16), title=dict(font=dict(size=16),
                                                                                                   side='top'),
                                                            )
                      )
    return fig


def pourcent2(abscisse, ordonnee, dataf, codes, legendtitle='', xaxis=''):
    agg2 = dataf[[abscisse, ordonnee]].groupby(by=[abscisse, ordonnee]).aggregate({abscisse: 'count'}).unstack().fillna(
        0)
    agg = agg2.T / agg2.T.sum()
    agg = agg.T.round(2) * 100
    if abscisse == 'Village_clean':
        agg = agg.reindex(
            ['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood", 'Al-Samoud'])
        agg2 = agg2.reindex(
            ['Bit Boos', "Old Sana'a", "Enma'a", "Alkatea'a", "Hada'a", 'AlGamea', "Alomall neighborhood", 'Al-Samoud'])
    x = agg.index
    x = agg2.index
    if ordonnee.split(' ')[0] in codes['list name'].values:
        colors_code = codes[codes['list name'] == ordonnee.split(' ')[0]].sort_values(['code']).copy()
        labels = colors_code['label'].tolist()
        colors = colors_code['color'].tolist()
        fig = go.Figure()
        for i in range(len(labels)):
            if labels[i] in dataf[ordonnee].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse, labels[i])], name=labels[i], marker_color=colors[i].lower(),
                                     customdata=agg2[(abscisse, labels[i])], textposition="inside",
                                     texttemplate="%{customdata} persons", textfont_color="black")
                              )
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:, 0], name=agg.columns.tolist()[0][1], marker_color='green',
                               customdata=agg2.iloc[:, 0], textposition="inside", texttemplate="%{customdata} persons",
                               textfont_color="black")
                        )
        for i in range(len(agg.columns) - 1):
            fig.add_trace(
                go.Bar(x=x, y=agg.iloc[:, i + 1], name=agg.columns.tolist()[i + 1][1], customdata=agg2.iloc[:, i + 1],
                       textposition="inside", texttemplate="%{customdata} persons", textfont_color="black")
                )
    fig.update_layout(barmode='relative', xaxis={'title': xaxis, 'title_font': {'size': 18}},
                      yaxis={'title': 'Percentages', 'title_font': {'size': 18}}
                      )
    fig.update_layout(legend_title=legendtitle, legend=dict(orientation='h', yanchor="bottom", y=1.02, xanchor="center",
                                                            x=0.5, font=dict(size=16), title=dict(font=dict(size=16),
                                                                                                   side='top'),
                                                            )
                      )
    return fig

def show_data(row,df,codes,categs=None):
    st.subheader(row['title'])
    if row['graphtype'] == 'treemap':
        # fig=go.Figure()
        # fig.add_trace(go.Treemap(branchvalues='total',labels=data[quest.iloc[i]['variable_x']],parents=data[quest.iloc[i]['variable_y']],
        #			  root_color="lightgrey",textinfo="label+value"))
        # st.write(df)
        fig = px.treemap(df, path=[row['variable_x'], row['variable_y']],
                         values='persons', color=row['variable_y'])
        st.plotly_chart(fig, use_container_width=True)

    elif row['graphtype'] == 'violin':
        fig = go.Figure()
        if categs == None:
            if row['variable_x'].split(' ')[0] in codes['list name'].unique():
                categs = codes[codes['Id'] == row['variable_x'].split(' ')[0]].\
                    sort_values(by='code')['label'].tolist()
            else:
                categs = df[row['variable_x']].unique()

        for categ in categs:
            fig.add_trace(go.Violin(x=df[row['variable_x']][df[row['variable_x']] == str(categ)],
                                    y=df[row['variable_y']][df[row['variable_x']] == str(categ)],
                                    name=categ,
                                    box_visible=True,
                                    meanline_visible=True, points="all", ))

        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[-0.1, df[row['variable_y']].max() + 1])
        fig.update_layout(yaxis={'title': row['ytitle'], 'title_font': {'size': 18}})

        st.plotly_chart(fig, use_container_width=True)
	
    elif row['graphtype'] == 'bar':
        # st.write(df[quest.iloc[i]['variable_y']].dtype)
        col1, col2 = st.columns([1, 1])
        fig1 = count2(row['variable_x'], row['variable_y'],
                      df, codes, legendtitle=row['legendtitle'], xaxis=row['xtitle'])
        col1.plotly_chart(fig1, use_container_width=True)
        fig2 = pourcent2(row['variable_x'], row['variable_y'],
                         df,codes, legendtitle=row['legendtitle'], xaxis=row['xtitle'])
        col2.plotly_chart(fig2, use_container_width=True)

    elif row['graphtype'] == 'map':
        fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color=row['variable_x'],
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    
    elif row['graphtype'] == 'radar':
        col1,col2= st.columns([1,1])
        items=df[row['variable_y']].unique().tolist()
        
        for i in range(len(items)):
            
            categories = [' '.join((i.split(' ')[1:])) for i in df if row['variable_x'] in i[:len(row['variable_x'])]]
            
            r_categ=df[df[row['variable_y']]==items[i]][[i for i in df if row['variable_x'] in i[:len(row['variable_x'])]]].mean()
            r_all=df[[i for i in df if row['variable_x'] in i[:len(row['variable_x'])]]].mean()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=r_categ, theta=categories, fill='toself', name=items[i]))
            
            fig2.add_trace(go.Scatterpolar(r=r_all, theta=categories, fill='toself', name='All dataset'))
            
            fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, max(r_all.max(),r_categ.max())])),showlegend=True)

            
            
            
            
            if i%2==0:
            	col1.plotly_chart(fig2,use_container_width=True)
            else:
                col2.plotly_chart(fig2,use_container_width=True)
    
    
    
    
    
    st.write(row['Description'])

def select_data(row,data,cat_cols):

    if row['variable_x'] in cat_cols:
        if row['graphtype'] == 'map':
            cat = row['variable_x']
            df = pd.DataFrame(columns=[cat, 'latitude', 'longitude'])
            catcols = [j for j in data.columns if cat in j]
            cats = [' '.join(i.split(' ')[1:]) for i in catcols]
            for n in range(len(catcols)):
                ds = data[[catcols[n], 'latitude', 'longitude']].copy()
                ds = ds[ds[catcols[n]].isin(['Yes', 1])]
                ds[catcols[n]] = ds[catcols[n]].apply(lambda x: cats[n])
                ds.columns = [cat, 'latitude', 'longitude']
                df = df.append(ds)
            df['persons'] = np.ones(len(df))
            return df

        elif row['variable_y'] in cat_cols:
            df = pd.DataFrame(columns=[row['variable_x'], row['variable_y']])
            quests1 = [i for i in data.columns if row['variable_x'] in i]
            catq1 = [' '.join(i.split(' ')[1:]) for i in quests1]

            # st.write(quests1)
            quests2 = [i for i in data.columns if row['variable_y'] in i]
            catq2 = [' '.join(i.split(' ')[1:]) for i in quests2]
            # st.write(quests2)
            # st.write(dfm[quests1+quests2])
            for cat_x in range(len(quests1)):
                for cat_y in range(len(quests2)):
                    ds = data[[quests1[cat_x], quests2[cat_y]]].copy()
                    ds = ds[ds[quests1[cat_x]].isin(['Yes', 1])]
                    ds = ds[ds[quests2[cat_y]].isin(['Yes', 1])]
                    ds[quests1[cat_x]] = ds[quests1[cat_x]].apply(lambda x: catq1[cat_x])
                    ds[quests2[cat_y]] = ds[quests2[cat_y]].apply(lambda x: catq2[cat_y])
                    ds.columns = [row['variable_x'], row['variable_y']]
                    df = df.append(ds)
            df['persons'] = np.ones(len(df))
            return df

    if row['variable_x'] in cat_cols or row['variable_y'] in cat_cols:
        df=data.copy()
        if row['variable_x'] in cat_cols:
            cat, autre = row['variable_x'], row['variable_y']
        else:
            cat, autre = row['variable_y'], row['variable_x']
        catcols = [j for j in data.columns if cat in j[:len(cat)]]
        
        df['persons'] = np.ones(len(df))
        return df[['persons',autre]+catcols]

    elif row['graphtype'] == 'map':
        df = data[[row['variable_x'],'latitude','longitude']].copy()
        return df

    else:
        df = data[[row['variable_x'], row['variable_y']]].copy()
        df['persons'] = np.ones(len(df))
        return df

