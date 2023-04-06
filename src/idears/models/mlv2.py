
import numpy as np
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import os


import shap
#from boruta import BorutaPy
from xgboost import XGBClassifier,plot_importance
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

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
		
		#Variable map
		# *todo - move to separate file
		self.variablemap=dict({'testosterone_f30850_0_0':'Testosterone',
        'age_when_attended_assessment_centre_f21003_0_0':'Age at baseline',
        'parental_pd':'Parent with PD',
        'neutrophill_percentage_f30200_0_0':'Neutrophill percentage',
        'hdl_cholesterol_f30760_0_0':'HDL Cholesterol',
        'igf1_f30770_0_0':'IGF1',
        'suffer_from_nerves_f2010_0_0':'Suffer from nerves',
        'avg_duration_to_first_press_of_snapbutton_in_each_round':'Avg duration to first press - snap button',
        'neutrophill_lymphocyte_ratio':'Neutrophill:Lymphocyte count ratio',
        'creactive_protein_f30710_0_0':'C-reactive protein',
        'Retired':'Retired at baseline',
        'triglycerides_f30870_0_0':'Triglycerides',
        'creatinine_enzymatic_in_urine_f30510_0_0':'Creatinine enzymatic in urine',
        'total_bilirubin_f30840_0_0':'Bilirubin',
        'cholesterol_f30690_0_0':'Cholesterol',
        'apolipoprotein_a_f30630_0_0':'Apoplipoprotein A',
        'glycated_haemoglobin_hba1c_f30750_0_0':'Glycated haemoglobin',
        'creatinine_f30700_0_0':'Creatine',
        'vitamin_d_f30890_0_0':'Vitamin D',
        'platelet_crit_f30090_0_0':'Platelet crit',
        'number_of_treatmentsmedications_taken_f137_0_0':'# of treatments/ medications',
        'hip_circumference_f49_0_0':'Hip circumference',
        'usual_walking_pace_f924_0_0':'Usual walking pace',
        'AST_ALT_ratio':'AST:ALT ratio',
        'Total ICD10 Conditions at baseline':'Total ICD10 Conditions at baseline',
        'waist_circumference_f48_0_0':'Waist circumference',
        'sex_f31_0_0':'Gender',
        'forced_vital_capacity_fvc_f3062_0_0':'Forced vital capacity',
        'standing_height_f50_0_0':'Height',
        'mean_reticulocyte_volume_f30260_0_0':'Mean reticulocyte volume',
        'hand_grip_strength_left_f46_0_0':'Hand grip strength (left)',
        'lymphocyte_count_f30120_0_0':'Lymphocyte count',
        'chest_pain_or_discomfort_f2335_0_0':'Chest pain or discomfort',
        'platelet_count_f30080_0_0':'Platelet count',
        'alanine_aminotransferase_f30620_0_0':'Alanine aminotransferase',
        'hand_grip_strength_right_f47_0_0':'Hand grip strength (right)',
        'ldl_direct_f30780_0_0':'LDL Direct',
        'neutrophill_count_f30140_0_0':'Neutrophill Count',
        'number_of_selfreported_noncancer_illnesses_f135_0_0':'Number of self reported non cancer illnesses',
        'urate_f30880_0_0':'Urate',
        'coffee_intake_f1498_0_0':'Coffee Intake',
        'depressed':'Depression',
        'hypertension':'Hypertension',
        'ibuprofen':'Taking Ibuprofen',
        'ipaq_activity_group_f22032_0_0':'IPAQ Activity Level',
        'mean_corpuscular_volume_f30040_0_0':'Mean corpuscular volume',
        'mother_still_alive_f1835_0_0':'Mother Still Alive',
        'neuroticism_score_f20127_0_0':'Neuroticism Score',
        'non_ost':'Non-steroidal anti-inflammatories',
        'non_ost_non_asp':'Non-steroidal anti-inflammatories excluding aspirin',
        'smoking_status_f20116_0_0':'Smoking',
        'urban_rural':'Urban Rural Score',
        'taking_other_prescription_medications_f2492_0_0':'Taking other prescription medications',
        'inverse_distance_to_the_nearest_major_road_f24012_0_0':'Inverse distance to nearest main road',
        'mean_time_to_correctly_identify_matches_f20023_0_0':'Mean time to correctly identify matches',
        'nervous_feelings_f1970_0_0':'Nervous feelings',
        'phosphate_f30810_0_0':'Phosphate',

        #added for dementia shap charts 2/3/22
        'longstanding_illness_disability_or_infirmity_f2188_0_0':'Longstanding illness, disability, infirmity',
        'average_total_household_income_before_tax_f738_0_0':'Average hhold income before tax',
        'overall_health_rating_f2178_0_0':'Overall health rating', 'perc_correct_matches_rounds':'Percentage correct rounds',
        'sleeplessness_insomnia_f1200_0_0':'Sleeplessness/ insomina',
        'time_spent_driving_f1090_0_0':'Time spent driving',
        'processed_meat_intake_f1349_0_0':'Processed meat intake',
        'falls_in_the_last_year_f2296_0_0':'Falls in last year', 'urea_f30670_0_0':'Urea',
        'frequency_of_depressed_mood_in_last_2_weeks_f2050_0_0':'Frequency of depressed mood in last week',
        'number_of_incorrect_matches_in_round_f399_0_2':'Number of incorrect matches in round',
        'frequency_of_unenthusiasm_disinterest_in_last_2_weeks_f2060_0_0':'Frequency of unenthusiasm/ disinterest',
        'never_eat_eggs_dairy_wheat_sugar_f6144_0_0_I eat all of the above':'Eat all eggs, wheat, dairy',
        'number_of_incorrect_matches_in_round_f399_0_1':'Number of incorrect matches in round',
        'worry_too_long_after_embarrassment_f2000_0_0':'Worry too long after embarrassement',
        'cystatin_c_f30720_0_0':'Cystatin',
        'lymphocyte_percentage_f30180_0_0':'Lymphocyte percentage',
        'red_blood_cell_erythrocyte_count_f30010_0_0':'Red blood cell erythrocyte count',
        'sedentary_time':'Sedentary time'}) 

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

	def auc_score(self,valid_x,valid_y,model,mod_name='XGB'):
		pred=model.predict_proba(valid_x)[:, 1]
		score = roc_auc_score(valid_y,pred)
		print('AUC '+mod_name+': ',str(score))
		return score