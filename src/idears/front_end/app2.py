# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:17:07 2021
@author: Mike Allwright
"""
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numerize import numerize

import sys
#change sys path based on where code sits for you
sys.path.append('/Users/michaelallwright/Documents/github/ukb/codebase1/src/idears/')

from preprocessing.idears_backend import *
from results.charts import charts

ib=Idears()
ch=charts()


st.set_page_config(page_title='IDEARS Pipeline',  layout='wide', page_icon=':machine learning:')

#this is the header

t1, t2 = st.columns((0.001,1)) 

t2.title("IDEARs")
t2.markdown(" **tel:** 07450554693 **| website:** https://www.allwrightanalytics.org **| email:** mailto: michael@allwrightanalytics.com")


#t1.image('images/index.png', width = 120)
#t2.title("Customer Analytics and Optimisation Reports")

## Data

app_mode = st.sidebar.selectbox("Choose the app mode",
["Population Characteristics", "SHAP Analysis", "Variable level comparisons"])



with st.spinner('Updating Report...'):

    if app_mode=="SHAP Analysis":
        t1, t2 = st.columns((0.001,1)) 

        t2.header("SHAP Analysis - a proxy for the most important features driving the analysis")

        feats_full=pd.read_csv(ib.path+'feats_full.csv')
        df_feats_sum=pd.read_csv(ib.path+'df_feats_sum.csv')
        df_auc=pd.read_csv(ib.path+'df_auc.csv')
        df_avg_vals=pd.read_csv(ib.path+'df_avg_vals.csv')

        m2, m3, m4, m5,m6 = st.columns((1,1,1,1,1))

        #st.write(feats_full.head())

        df_feats_sum['disease']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[0])
        df_feats_sum['age']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[1])
        df_feats_sum['gender']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[2])
        df_feats_sum['APOE4']=df_feats_sum['breakdown'].apply(lambda x:x.split('|')[3])


        diseases=st.selectbox("Choose which diseases to model",list(df_feats_sum['disease'].unique()))
        ages=st.selectbox("Choose which age ranges to model",list(df_feats_sum['age'].unique()))
        genders=st.selectbox("Choose which age ranges to model",list(df_feats_sum['gender'].unique()))
        apoes=st.selectbox("Choose which apoes to model",list(df_feats_sum['APOE4'].unique()))

        #determine what the breakdown selected is for filter below
        bdown='|'.join([diseases,ages,genders,apoes])

        #st.write(bdown)

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

        var_sel=st.selectbox("Select Variable to Compare",list(df_avg_vals['Attribute'].unique()))
        
       
       
        df_avg_vals_sum=pd.melt(df_avg_vals,id_vars=['breakdown','Attribute','p value'],value_vars=['case_mean','ctrl_mean'])
        mask=(df_avg_vals_sum['breakdown']==bdown)&(df_avg_vals_sum['Attribute']==var_sel)
        df_avg_vals_sum=df_avg_vals_sum.loc[mask,]

        st.write("Summary for "+var_sel)
        pval=df_avg_vals_sum['p value'].mean()
        if pval<0.001:
            pval="statistically significant (p<0.001)"
        elif pval<0.01:
            pval="statistically significant (p<0.01)"
        elif pval<0.05:
            pval="statistically significant (p<0.05)"
        else:
            pval="not significant"
        st.write(var_sel+" is "+pval+" for comparing "+diseases+" to control")
        fig = px.bar(df_avg_vals_sum, x='value', y='variable')
        g1,g2 = st.columns((1,1))
        g1.plotly_chart(fig, use_container_width=True) 
       


        

