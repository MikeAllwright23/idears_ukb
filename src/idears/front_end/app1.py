

# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving	

import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

import sys
sys.path.append('/Users/michaelallwright/Documents/github/ukb/pipeline/')
from data_gen import *
from data_proc import *
from ml import *

di=data_import()
dp=data_proc()
ml=ml_funcs()


tick_vals_dic={'Carriers':[0, 0.1, 0.2, 0.3, 0.4],'Non Carriers':[0, 0.05, 0.1, 0.15, 0.2],
				'All':[0, 0.1, 0.2, 0.3, 0.4]}

tickvals=[0, 0.1, 0.2, 0.3, 0.4]
print_all=True

@st.cache
def set_data(depvar,icd10s):
	df=di.create_model_data(df=None,import_parquet=True,depvar=depvar,icd10s=icd10s,infile='ukb_df_processed.parquet')
	return df

@st.cache
def import_run():
	df_tr=pd.read_parquet(di.path+"AD/df_train_20221116.parquet")
	df_te=pd.read_parquet(di.path+"AD/df_test_20221116.parquet")
	dropcols=['fasting_time_f74_0_0']
	df_train1=ml.data_process(df_tr)[0]
	df_train1.drop(columns=dropcols,inplace=True)
	df=pd.DataFrame([])


	dict_apoe={"All":[0,1],"Non Carriers":[0],"Carriers":[1]}

	for a in dict_apoe:
	    mask=(df_train1['APOE4_Carriers'].isin(dict_apoe[a]))
	    df_traina=df_train1.loc[mask,]
	    df_dict1_a=dp.split_mult_files(df=df_traina,depvar="AD",
	             normvars=['age_when_attended_assessment_centre_f21003_0_0'],iterations=2,mult_fact_max=1)
	    for j in range(len(df_dict1_a)):
	        mask=(df_traina['eid'].isin(df_dict1_a[j]))
	        df_train2a=df_traina.loc[mask,]
	        #df_train2=ml.data_process(df_train2)[0]

	        feats_all=ml.get_shap_feats(df_train2a.drop(columns='eid'))
	        feats_all['iteration']=j
	        feats_all['Disease']=a
	        #feats_all['APOE']=apo
	        df=pd.concat([df,feats_all],axis=0)
	df.sort_values(by='mean_shap',ascending=False,inplace=True)

	return df


def file_selector(folder_path='data/'):
	filenames = os.listdir(folder_path)
	selected_filename = st.selectbox('Select a file', filenames)
	return os.path.join(folder_path, selected_filename)


st.title('UK Biobank Machine Learning by disease')


icd10s=st.text_input('Enter ICD10s')
depvar=st.text_input('Enter name for model variable')

imp_files_option=False
imp_files_option = st.selectbox('Import Files',
[False,True],index=0)

if imp_files_option:
	df1=set_data(icd10s=icd10s,depvar=depvar)
	df=import_run()

else:
	filename = file_selector()
	#st.write('You selected `%s`' % filename)

	df=pd.read_csv(filename)
	df.sort_values(by='mean_shap',ascending=False,inplace=True)

diseases=[None]+list(df['Disease'].unique())


#st.write(diseases)

option = st.selectbox('Select Disease',
diseases,index=1)

recs1 = st.slider('Number of features to display', 0, 130, 25)
#st.write(recs)
dis_vars=['All']+[None]+list(df['Attribute'].unique())

options = st.multiselect(
	'Which variables would you like to model',dis_vars)

st.write('You selected:', options)

if "All" in options:
	options = list(df['Attribute'].unique())


mask=df['Attribute'].isin(options)
df=df.loc[mask,]
#st.write('You selected:', option)

app_mode = st.sidebar.selectbox("Choose the app mode",
["Show instructions", "Run the app", "Show the source code"])

def read_markdown_file(markdown_file):
	return Path(markdown_file).read_text()

if app_mode=="Show instructions":
	intro_markdown = read_markdown_file("instructions.md")
	st.markdown(intro_markdown, unsafe_allow_html=True)
	

def run_the_app():
	#run the app in here.
	return None


#print(df.head())

def make_econ_bar(df,sort_var='mean_shap',err_var=None,recs=recs1,title='APOE4 Carriers Feature Importance',
	sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
	tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2],shrink=True,y_max=recs1-0.5,figsize=(3,16),out=True):

	 # Setup plot size.

	figsize=(3,1+recs/4)
	fig, ax = plt.subplots(figsize=figsize)

	# Create grid 
	# Zorder tells it which layer to put it on. We are setting this to 1 and our data to 2 so the grid is behind the data.
	ax.grid(which="major", axis='x', color='#758D99', alpha=0.6, zorder=1)

	# Remove splines. Can be done one at a time or can slice with a list.
	ax.spines[['top','right','bottom']].set_visible(False)

	# Make left spine slightly thicker
	ax.spines['left'].set_linewidth(1.1)
	ax.spines['left'].set_linewidth(1.1)

	# Setup data
	df_bar = df.sort_values(by=sort_var,ascending=False).head(recs)
	st.write(df_bar.head())
	df_bar=df_bar.sort_values(by=sort_var,ascending=True)
	#st.write(df_bar)
	
	custom_palette=['midnightblue' if c<0 else '#E3120B' for c in list(df_bar['corr'])]#'#006BA2'


	# Plot data here
	# plot error bar if specified based on error variable name
	if err_var is None:
		ax.barh(df_bar['Attribute'], df_bar[sort_var].round(3), color=custom_palette, zorder=2)#
	else:
		ax.barh(df_bar['Attribute'], df_bar[sort_var].round(3), color=custom_palette, zorder=2,xerr=df_bar[err_var])#

	# Set custom labels for x-axis
	ax.set_xticks(tick_vals_use)
	ax.set_xticklabels(tick_vals_use)

	# Reformat x-axis tick labels
	ax.xaxis.set_tick_params(labeltop=True,      # Put x-axis labels on top
							 labelbottom=False,  # Set no x-axis labels on bottom
							 bottom=False,       # Set no ticks on bottom
							 labelsize=11,       # Set tick label size
							 pad=-1)             # Lower tick labels a bit

	# Reformat y-axis tick labels
	ax.set_yticklabels(df_bar['Attribute'],      # Set labels again
					   ha = 'left')              # Set horizontal alignment to left
	ax.yaxis.set_tick_params(pad=300,            # Pad tick labels so they don't go over y-axis
							 labelsize=11,       # Set label size
							 bottom=False)       # Set no ticks on bottom/left

	# Shrink y-lim to make plot a bit tighter
	if shrink:
		ax.set_ylim(-0.5,y_max )

	# Add in line and tag
	ax.plot([-1.30, .87],                 # Set width of line
			[1.02, 1.02],                # Set height of line
			transform=fig.transFigure,   # Set location relative to plot
			clip_on=False, 
			color='midnightblue',#E3120B', 
			linewidth=.6)
	ax.add_patch(plt.Rectangle((-1.30,1.02),                # Set location of rectangle by lower left corder
							   0.22,                       # Width of rectangle
							   -0.02,                      # Height of rectangle. Negative so it goes down.
							   facecolor='midnightblue',#'#E3120B', 
							   transform=fig.transFigure, 
							   clip_on=False, 
							   linewidth = 0))

	# Add in title and subtitle
	ax.text(x=-1.30, y=.96, s=title, transform=fig.transFigure, ha='left', fontsize=13, weight='bold', alpha=.8)
	ax.text(x=-1.30, y=.925, s=sub_title, transform=fig.transFigure, ha='left', fontsize=11, alpha=.8)

	# Set source text
	ax.text(x=-1.30, y=.08, s=footer, transform=fig.transFigure, ha='left', fontsize=9, alpha=.7)
	
	if labels_show:
		for bars in ax.containers:
			ax.bar_label(bars)

	if out:
		# Export plot as high resolution PNG
		plt.savefig(outfile,    # Set path and filename
					dpi = 300,                     # Set dots per inch
					bbox_inches="tight",           # Remove extra whitespace around plot
					facecolor='white')  

	return fig,df_bar           # Set background color to white

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.

foot="""Source: UK Biobank. Red indicates associated variable, blue indicates protective"""
foort=""""""

if print_all:

	df_all=pd.DataFrame([])
	for p in df['Disease'].unique():
		mask=(df['Disease']==p)
		#df1=pd.DataFrame(df.loc[mask,].groupby('Attribute')['mean_shap','corr'].mean()).reset_index()

		df1=pd.DataFrame(df.loc[mask,].groupby('Attribute').agg({'mean_shap':['mean','std'],'corr':'mean'})).reset_index()
		df1.columns=['Attribute','mean_shap','std_shap','corr']

		
		fig,df_bar =make_econ_bar(df1,sort_var='mean_shap',err_var='std_shap',recs=recs1,title='Top Features for '+p,shrink=True,
		  sub_title="Mean SHAP Score",footer=foot,
		  outfile=p+'chart.png',
						 tick_vals_use=tickvals)

		df_bar['Disease']=p

		df_all=pd.concat([df_all,df_bar],axis=0)

		st.pyplot(fig)

else:
	mask=(df['Disease']==option)
	df1=pd.DataFrame(df.loc[mask,].groupby('Attribute')['mean_shap','corr'].mean()).reset_index()

	

	fig,df_bar=make_econ_bar(df1,sort_var='mean_shap',err_var='std_shap',recs=recs1,title='Top Features for '+option,
			 sub_title="Mean SHAP Score",footer=foot
			 ,outfile=option+'chart.png',
				 tick_vals_use=tickvals)
	df_all=df_bar.copy()
	st.pyplot(fig)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_all)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='disease_shap.csv',
    mime='text/csv',
)

#st.button("Re-run")