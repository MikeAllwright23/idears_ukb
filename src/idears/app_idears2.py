# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:17:07 2021
@author: Mike Allwright
"""

path='src/idears/'#/Users/michaelallwright/Documents/github/ukb/codebase1/src/idears/'
import sys
sys.path.append(path)
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numerize import numerize

import os

from preprocessing.idears_backend import *
from results.charts import charts

ib=Idears(path=path)
ch=charts()


st.set_page_config(page_title='IDEARS Pipeline',  layout='wide', page_icon=':machine learning:')

#this is the header

t1, t2 = st.columns((0.001,1)) 

t2.title("IDEARs: Integrated Disease Explanation and Risk Scoring Platform")
t2.markdown(" **tel:** 07450554693 **| website:** https://www.allwrightanalytics.org **| email:** mailto: michael@allwrightanalytics.com")
t2.markdown("The Forefront Research Group, University of Sydney https://www.forefrontresearch.org/bmc-bioinformatics-and-statistics/ ")

#t1.image('images/index.png', width = 120)
#t2.title("Customer Analytics and Optimisation Reports")

## Data
from pathlib import Path
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

app_mode = st.sidebar.selectbox("Choose the app mode",
["Introduction","SHAP Analysis"])

@st.cache(allow_output_mutation=True)
def load_file(file):
	return pd.read_csv(ib.path+file)


with st.spinner('Updating Report...'):  

    if app_mode=="Introduction":
        import os
       
        intro_markdown = read_markdown_file(path+"README.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)


    if app_mode=="SHAP Analysis":
        t1, t2 = st.columns((0.001,1)) 

        t2.header("SHAP Analysis - : The most important features in predicting a given disease")

        st.write("In the below section, select the specific cohort attributes you would like to model.")

        #Import all local files output through IDEARs backend for selection and graphical display
        feats_fullx=load_file('feats_full.csv')#pd.read_csv(ib.path+'feats_full.csv')
        feats_full=feats_fullx.copy()
        df_feats_sumx=load_file('df_feats_sum.csv')#pd.read_csv(ib.path+'df_feats_sum.csv')
        df_feats_sum=df_feats_sumx.copy()
        df_aucx=load_file('df_auc.csv')#pd.read_csv(ib.path+'df_auc.csv')
        df_auc=df_aucx.copy()
        df_avg_valsx=load_file('df_avg_vals.csv')#pd.read_csv(ib.path+'df_avg_vals.csv')
        df_avg_vals=df_avg_valsx.copy()

        #st.write(feats_full.head())

        df_feats_sum['disease']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[0])
        df_feats_sum['age']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[1])
        df_feats_sum['gender']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[2])
        df_feats_sum['APOE4']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[3])

        #User select boxes to determine which of a number of variables to select

        t1, t2 = st.columns((1,1)) 
        diseases=t1.selectbox("Choose which diseases to model",list(df_feats_sum['disease'].unique()),
                              help='Select the disease Alzheimers Disease: AD, Parkinsons: PD etc.')
        ages=t2.selectbox("Choose which age ranges to model",list(df_feats_sum['age'].unique()),
              
                          help='Filter the cohort for given ages at baseline')
        t1, t2 = st.columns((1,1)) 
        genders=t1.selectbox("Choose which genders to model",list(df_feats_sum['gender'].unique()),
                             help='Filter the cohort for given genders at baseline')
        apoes=t2.selectbox("Choose which apoes to model",list(df_feats_sum['APOE4'].unique()),
                           help='As APOE4 Carriers are deemed important for neurodegnerative diseases,\
                             select whether carriers or not or all here')

        #determine what the breakdown selected is for filter below
        bdown='|'.join([diseases,ages,genders,apoes])

        #Filter dataframe by breakdown
        mask=(df_feats_sum['breakdown']==bdown)
        df_s=df_feats_sum.loc[mask,]

        #st.write(df_s)

        #Display output 
        max_shap=df_feats_sum['mean_shap'].max()
        if max_shap>1:
            tick_vals_use=[0, 0.2, 0.4, 0.6, 0.8, 1]

        elif max_shap>0.25:
            tick_vals_use=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2]

        fig,df_bar=ch.make_econ_bar(df_s,sort_var='mean_shap',err_var=None,recs=None,title='APOE4 Carriers Feature Importance',
		sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
		tick_vals_use=tick_vals_use,shrink=True,figsize=(3,16),out=True)

        st.pyplot(fig)

        #SHow scatterplot

        var_compare=st.selectbox("Choose which variable to compare",['APOE4','gender'])
        g1,g2 =st.columns((5,5))

        
        g1.header("Scatterplot to show Mean SHAP Variables by "+var_compare)
        

        if var_compare=='gender':
            mask=(df_feats_sum['disease']==diseases)&(df_feats_sum['age']==ages)&(df_feats_sum['APOE4']==apoes)
            df_feats_sum_apoe=pd.DataFrame(df_feats_sum.loc[mask,].groupby(['Attribute',var_compare])['mean_shap'].mean().unstack(var_compare)).reset_index()
        elif var_compare=='APOE4':
            mask=(df_feats_sum['disease']==diseases)&(df_feats_sum['age']==ages)&(df_feats_sum['gender']==genders)
            df_feats_sum_apoe=pd.DataFrame(df_feats_sum.loc[mask,].groupby(['Attribute',var_compare])['mean_shap'].mean().unstack(var_compare)).reset_index()
           

        var1=list(df_feats_sum_apoe.columns)[2]
        var2=list(df_feats_sum_apoe.columns)[3]
        fig=px.scatter(df_feats_sum_apoe,x=df_feats_sum_apoe[var1],y=df_feats_sum_apoe[var2],hover_data=['Attribute'])
        #g1.plotly_chart(fig, use_container_width=True)

        df_feats_sum_apoe['tot']=df_feats_sum_apoe.apply(lambda x:x[var1]+x[var2],axis=1)
        df_feats_sum_apoe2=df_feats_sum_apoe.sort_values(by='tot',ascending=False).head(25).sort_values(by='tot',ascending=True)

        fig,ax=ch.dumbell_plot(df=df_feats_sum_apoe2,att_var="Attribute",var1=var1,var2=var2)
        st.pyplot(fig)
        #g2.plotly_chart(st.pyplot(fig), use_container_width=True)

        st.header("Comparing Individual Variables of Interest for the Selections Provided:")
        #change field names for analysis
        field_map=ib.field_map()
        mask=pd.notnull(df_avg_vals['Attribute'].map(field_map))
        df_avg_vals.loc[mask,'Attribute']=df_avg_vals.loc[mask,'Attribute'].map(field_map)

        var_sel=st.selectbox("Select Variable to Compare",list(df_avg_vals['Attribute'].unique()))
        df_avg_vals['disease']=df_avg_vals['breakdown'].apply(lambda x:x.split('|')[0])
        
        df_avg_vals['age']=df_avg_vals['breakdown'].apply(lambda x:x.split('|')[1])
        df_avg_vals['gender']=df_avg_vals['breakdown'].apply(lambda x:x.split('|')[2])
        df_avg_vals['APOE4']=df_avg_vals['breakdown'].apply(lambda x:x.split('|')[3])

        #mask=(df_avg_vals['disease']==diseases)&(df_avg_vals['age']==ages)&(df_avg_vals['gender']==genders)
        #df_avg_vals_sum=pd.DataFrame(df_avg_vals.loc[mask,].groupby(['Attribute',var_compare])['case_mean','ctrl_mean'].mean()).reset_index()#.unstack(var_compare)

        #st.write(df_avg_vals_sum)

        df_avg_vals_sum=pd.melt(df_avg_vals,id_vars=['disease','age','gender','APOE4','Attribute','p value'],value_vars=['case_mean','ctrl_mean'])
        #mask=(df_avg_vals_sum['Attribute'].str.contains('igf1'))
        #df_avg_vals_sum=df_avg_vals_sum.loc[mask,]

        df_avg_vals_sum['pval']=df_avg_vals_sum['p value'].apply(lambda x:"(***)" if x<0.001 else ("(**)" if x<0.01 else "(*)" if x<0.05 else "ns"))

        #st.write(df_avg_vals_sum)

       
        #df_avg_vals_sum=pd.melt(df_avg_vals,id_vars=['breakdown','Attribute','p value'],value_vars=['case_mean','ctrl_mean'])
        mask=(df_avg_vals_sum['Attribute']==var_sel)&(df_avg_vals_sum['disease']==diseases)&(df_avg_vals_sum['age']==ages)&(df_avg_vals_sum['gender']==genders)&(df_avg_vals_sum['APOE4']==apoes)#(df_avg_vals_sum['breakdown']==bdown)&
        df_avg_vals_sum=df_avg_vals_sum.loc[mask,]

        df_avg_vals_sum['corr']=df_avg_vals_sum['variable'].apply(lambda x:1 if x=='case_mean' else -1)

        #st.write("Summary for "+var_sel)
        pval=df_avg_vals_sum['p value'].mean()
        if pval<0.001:
            pval="statistically significant (p<0.001)"
        elif pval<0.01:
            pval="statistically significant (p<0.01)"
        elif pval<0.05:
            pval="statistically significant (p<0.05)"
        else:
            pval="not significant"
        
      
        title="In a comparison of "+var_sel+" between "+diseases+" cases and controls for age ranges "+str(ages)+" and genders "+str(genders)+" and APOE4 carrier types: "+str(apoes)+"."
        part_two="In this case "+var_sel+" is "+pval+" for comparing "+diseases+" to control."
        fig = px.bar(df_avg_vals_sum, y="variable", x="value", color="variable",orientation='h',text='value',title=title)

        fig.update_layout(
            title=title,
            xaxis_title="Value",
            xaxis = dict(
            tickfont = dict(size=20)),
            yaxis_title="",
            legend_title="Case vs Control",
            font=dict(
                family="Courier New, monospace",
                size=24,
                color="Black"
            )
        )
        #fig.show()
        
        #fig = px.bar(df_avg_vals_sum, x='value', y='variable',text_auto=True)
        fig.update_traces(textfont_size=30, textangle=0, textposition="inside", cliponaxis=True)

        st.subheader(title)
        st.subheader(part_two)

        df_avg_vals_sum2=df_avg_vals_sum.copy()
        df_avg_vals_sum2['Attribute']=df_avg_vals_sum.apply(lambda x:x['Attribute']+" "+x['variable'],axis=1)
        fig,ax=ch.make_econ_bar(df_avg_vals_sum2,sort_var='value',err_var=None,recs=3,title=var_sel,
		sub_title="",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
		tick_vals_use=None,shrink=False,figsize=(3,16),out=True,line=False)

        g1,g2 = st.columns((100,1))

        g1.pyplot(fig)
        #g1.plotly_chart(fig, use_container_width=True) 
       


        

