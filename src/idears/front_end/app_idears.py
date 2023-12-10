

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
from numerize import numerize


import sys
#change sys path based on where code sits for you
sys.path.append('/Users/michaelallwright/Documents/github/ukb/codebase1/src/idears/')
from preprocessing.idears_backend import *
from results.charts import charts
from d3blocks import D3Blocks
#
# Initialize
d3 = D3Blocks(chart='Chord', frame=False)
#
# Import example
df = d3.import_example('energy')
#
# Node properties
d3.set_node_properties(df, opacity=0.2, cmap='tab20')
d3.set_edge_properties(df, color='source', opacity='source')
#
# Show the chart
d3.show()
#
# Make some edits to highlight the Nuclear node
# d3.node_properties
d3.node_properties.get('Nuclear')['color']='#ff0000'
d3.node_properties.get('Nuclear')['opacity']=1
# Show the chart
#
d3.show()
# Make edits to highlight the Nuclear Edge
d3.edge_properties.get(('Nuclear', 'Thermal generation'))['color']='#ff0000'
d3.edge_properties.get(('Nuclear', 'Thermal generation'))['opacity']=0.8
d3.edge_properties.get(('Nuclear', 'Thermal generation'))['weight']=1000
#
# Show the chart
d3.show()

#st.pyplot(d3)



tickvals=[0, 0.1, 0.2, 0.3, 0.4]
print_all=True


st.set_page_config(page_title='SWAST - Handover Delays',  layout='wide', page_icon=':ambulance:')

#this i s the header
 

t1, t2 = st.columns((0.001,1)) 

t2.title('IDEARS: Integrated Disease Explanation And Risk Scoring Platform for the UKB:')
t2.markdown(" **tel:** +44 7450554693 **| website:** https://www.alchem-ai.org **| email:** mailto: michael@alchem-ai.org")


#t1.image('images/index.png', width = 120)
#t2.title("Customer Analytics and Optimisation Reports")

## Data

app_mode = st.sidebar.selectbox("Choose the app mode",
["Disease Risk Scoring", "Risk Factors", "Deep Dive"])



if app_mode=="Risk Factors":
	Expname='Hypertension Feature Importance SHAP - aged 50-60'
	fields=['All']
	ages=['50-60']
	gends=['All']
	diseases={'HT':['I10']}
	gend_dict_extcols={'All': ['age_when_attended_assessment_centre_f21003_0_0']}
	dis_exc_vars_dict={'HT':''}
	iters=2

	dis_dict={}

	m2, m3, m4, m5,m6 = st.columns((1,1,1,1,1))

	m2.Expname=st.text_input("Enter a description of the experiment you're running")
	m3.fields=[st.selectbox("Choose which fields to model",['All','Modifiable'])]

	dis = st.text_input("Enter a disease name")
	m4.icd10s = st.text_input("Enter icd10s separated with the | symbol")
	diseases=dict({dis:icd10s.split('|')})

	m5.ages=[st.selectbox("Choose which age ranges to model",['50-60','60-70','55-70'])]
	m6.gends=[st.selectbox("Choose which genders to model",['All','Male','Female'])]

	norms=[st.selectbox("Choose normalisation fields",['age_when_attended_assessment_centre_f21003_0_0'])]
	gend_dict_extcols={gends[0]: norms}

	excs=st.text_input("Choose field strings to exclude")
	dis_exc_vars_dict={dis:excs}



	@st.cache
	def run_models(fields,ages,gends,gend_dict_extcols,dis_exc_vars_dict,iters):
		# runs model steps 
		ib=idears()
		df_dict=ib.create_train_test(fields_include_use=fields,ages=ages,gends=gends,diseases=diseases)
		df_auc,feats_full=ib.get_aucs_all(df_dict,gend_dict_extcols=gend_dict_extcols,
		dis_exc_vars_dict=dis_exc_vars_dict,ages=ages,gends=gends,diseases=diseases,iters=iters)
		ch=charts()
		fig,df_bar=ch.make_econ_bar(df=feats_full,sort_var='mean_shap',err_var=None,recs=None,title=Expname,
			sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
			tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2],shrink=True,figsize=(3,16),out=True)
		

		return fig,df_bar,df_dict,df_auc,feats_full

	run='No'
	run=st.selectbox("Run",['No','Yes'])

	if run=="Yes":
		fig,df_bar,df_dict,df_auc,feats_full=\
	run_models(fields=fields,ages=ages,gends=gends,\
	gend_dict_extcols=gend_dict_extcols,dis_exc_vars_dict=dis_exc_vars_dict,iters=2)
		st.pyplot(fig)
		
