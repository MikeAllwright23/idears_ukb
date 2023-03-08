
import numpy as np
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import os


import shap
#from boruta import BorutaPy
from xgboost import XGBClassifier,plot_importance

from scipy.stats import norm

import seaborn as sns


class ml_funcs(object):

	def __init__(self):


        # import path
		self.path='/Users/michaelallwright/Documents/data/ukb/'

        #words to remove
		self.remwords=['Polymorphic','dementia','driving','eid','length_of_mobile_phone_use_f1110_0_0',\
'intercourse','job_code','date','records_in_hes','_speciality','death','sample_dilution','hospital_recoded','uk_biobank_assessment_centre',
'number_of_blood_samples_taken','ordering_of_blows','treatmentmedication_code','spirometry','carer_support_indicators',
'willing_to_attempt','patient_recoded','attendancedisability','cervical_smear_test']

		#xgb classifier model
		self.mod_xgb_base=XGBClassifier()

		#POS WEIGHT CHANGE
		self.mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree',  scale_pos_weight=1,colsample_bylevel=1,\
colsample_bynode=1, learning_rate=0.1,max_delta_step=0,  missing=1, n_estimators=60, n_jobs=4, \
nthread=4, objective='binary:logistic',random_state=0, reg_alpha=0, reg_lambda=1,\
min_child_weight=5,gamma=2, colsample_bytree=0.6,max_depth=5,seed=42, silent=None, subsample=1,\
verbosity=1,eval_metric='auc')

	def shap_sign(self,df,X):
		#df is the first SHAP dataset, X is the values dataset, find correlations between SHAP values and variables
		#values.

		corr = []
		for j in df.columns:
			#change to ensure nulls are filled in if they are there
			b = np.corrcoef(df[j].fillna(0),X[j].fillna(X[j].mean()))[1][0]
			corr.append(b)
			
		return corr


	def shap_out(self,sv,X,model):
		#returns SHAP abs df based on shap_values sv and includes correlations
		sv=sv.values
		df=pd.DataFrame(sv,columns=X.columns)
		corr=self.shap_sign(df,X)
		df_abs=abs(df)
		df_abs=pd.DataFrame(df_abs.mean(axis=0)).reset_index()
		df_abs.columns=['Attribute','mean_shap']
		df_abs['corr']=corr

		return df_abs

	def feat_imp(self,model,X):
		#returns built in feature importance for a given model with columns based on the trained dataset
		fi = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)),\
columns=['model_feature_importance','Attribute'])
		
		return fi

	def feats_out(self,sv,X,model):
		#combines feat imp and SHAP to return weighted breakdown
		
		df_shap=self.shap_out(sv,X,model)
		df=self.feat_imp(model,X)
		df=pd.merge(df_shap,df,on='Attribute',how='outer')

		df['weighted_shap']=df['mean_shap']/df['mean_shap'].sum()
		df['weighted_model_fi']=df['model_feature_importance']/df['model_feature_importance'].sum()
		df['shap_model_fi']=df['weighted_shap']+df['weighted_model_fi']
		mask=(df['mean_shap']>0)|(df['model_feature_importance']>0)
		df=df.loc[mask,]
		df.sort_values(by='shap_model_fi',ascending=False,inplace=True)
		
		return df

	def make_xy(self,df,depvar='AD'):
		predvars=[c for c in df.columns if c!=depvar and c!='eid']
		
		X=df[predvars]#feats_new_stu
		y=df[depvar]

		return X,y

	def fit_model(self,X,y):
		model=self.mod_xgb.fit(X, y)
		return model
	

	def get_shap_feats(self,df,depvar='AD'):
		
		X,y=self.make_xy(df,depvar=depvar)
		model=self.fit_model(X,y)

		explainer = shap.Explainer(model, X)
		shap_values = explainer(X,check_additivity=False)
		df_s=self.feats_out(shap_values,X,model)

		return df_s