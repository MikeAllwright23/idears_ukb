

import pandas as pd
import numpy as np
import ast
import re
import icd10
import datetime as dt

class data_proc(object):

	def __init__(self):	

		self.path='/Users/michaelallwright/Documents/data/ukb/'
		self.inpfile='ukb_df_processed.parquet'
		self.icd10_file='ukb_icd10s.parquet'

	

	#helper function to find columns in dataframe
	def findcols(self,df,string):
		return [col for col in df if re.search(string,col)]

	def melt_dis(self,df,val='disease'):
		#turn disease columns into rows applied to both diseases and their dates

		df = pd.melt(df, id_vars='eid', value_name=val)
		df=df[pd.notnull(df[val])]
		df.columns=['eid','variable',val]
		return df

	def split_disease_dfs(self,df):
		# create 2 dataframes, one for diseases and another for their corresponding dates

		cols_dis=[col for col in df.columns if '41270' in col or 'eid' in col]
		cols_date=[col for col in df.columns if '41280' in col or 'eid' in col]

		df_dis=df[cols_dis]
		df_date=df[cols_date]

		#replace variables so these can be merged later and then melt
		
		df_dis=self.melt_dis(df=df_dis,val='disease')
		df_dis['disease']=df_dis['disease'].str.replace("'","")
		df_dis['variable']=df_dis['variable'].str.replace('diagnoses_icd10_','')

		#replace variables so these can be merged later and format appropriately and melt
		df_date=self.melt_dis(df=df_date,val='dis_date')
		df_date['variable']=df_date['variable'].str.replace('41280','41270')
		df_date['variable']=df_date['variable'].str.replace('date_of_first_inpatient_diagnosis_icd10_','')
		
		df_date['dis_date']=df_date['dis_date'].str.replace('b','')
		df_date['dis_date']=df_date['dis_date'].str.replace("'","")
		df_date['dis_date']=pd.to_datetime(df_date['dis_date'])


		#remerge together
		df_dis_date=pd.merge(df_dis,df_date,on=['eid','variable'],how='left')

		#remerge to key variables for calculating disease times later
		df_dis_date=pd.merge(df_dis_date,df[['eid','date_of_attending_assessment_centre_f53_0_0']])

		df_dis_date['date_of_attending_assessment_centre_f53_0_0']=\
	pd.to_datetime(df_dis_date['date_of_attending_assessment_centre_f53_0_0'])

		df_dis_date['years_dis']=((df_dis_date['date_of_attending_assessment_centre_f53_0_0']-df_dis_date['dis_date']).dt.days/365.25)


		return df_dis_date
		

	def dis_list(self,df=None,icd10s=['G30']):
		#dependent variable
	
		icd10s='|'.join(icd10s)
		
		df=pd.read_parquet(self.path+self.icd10_file)

		df=self.split_disease_dfs(df)
		mask=((df['dis_date']-pd.to_datetime(df['date_of_attending_assessment_centre_f53_0_0'])).dt.days/365.25>2)
		df_dis_aft=df.loc[mask,]
		df_dis_bef=df.loc[~mask,]

		mask=(df_dis_aft['disease'].str.contains(icd10s,regex=True))
		dis_list_aft=list(df_dis_aft.loc[mask,'eid'].unique())

		mask=(df_dis_bef['disease'].str.contains(icd10s,regex=True))
		dis_list_already=list(df_dis_bef.loc[mask,'eid'].astype(str).unique())

		dis_list_out=[str(c) for c in dis_list_aft if c not in dis_list_already]
		
		return dis_list_out,dis_list_already


	def create_model_data(self,depvar='AD',icd10s=['G30'],infile='ukb_df_processed.parquet',
	nonull_var='date_of_all_cause_dementia_report_f42018_0_0'):
		
		# bring in processed data
		df=pd.read_parquet(self.path+infile)	
		dis_list_out,dis_list_already=self.dis_list(df=None,icd10s=icd10s)

		#exclude those who already had the disease at baseline or died and were in the control group 
		#or have a dementia related illness diagnosed and are in the control group
		cols=self.findcols(df,nonull_var)
		mask_exc=(((df['death']==1)|(df[cols].count(axis=1)>0))&~(df['eid'].isin(dis_list_out)))|\
	(df['eid'].isin(dis_list_already))

		df=df.loc[~mask_exc,]
		df[depvar]=0
		mask=(df['eid'].isin(dis_list_out))
		df.loc[mask,depvar]=1

		#any column with a value for dementia/ parkinsons here
		
		#
		return df

