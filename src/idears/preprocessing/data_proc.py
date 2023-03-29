

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
		dis_list_out,dis_list_already=self.dis_list(icd10s=icd10s)

		#exclude those who already had the disease at baseline or died and were in the control group 
		#or have a dementia related illness diagnosed and are in the control group
		cols=self.findcols(df,nonull_var)

		mask_exc=(((df['death']==1)|
		(df[cols].count(axis=1)>0))&~(df['eid'].isin(dis_list_out)))|\
	(df['eid'].isin(dis_list_already))

		df=df.loc[~mask_exc,]
		df[depvar]=0
		mask=(df['eid'].isin(dis_list_out))
		df.loc[mask,depvar]=1

		#any column with a value for dementia/ parkinsons here
		
		#
		return df


class normalisations():

	def __init__(self):	

		self.path='/Users/michaelallwright/Documents/data/ukb/'

	def case_control(self,df,depvar='AD'):
		"""
		split into case and control
		""" 
		mask=(df[depvar]==1)
		df_case=df.loc[mask,]
		df_ctrl=df.loc[~mask,]

		return df_case,df_ctrl

	def ctrl_case_ratios(self,df_case,df_ctrl,normvars):
		"""
		determine the ratio of control and case for each normvar denomination
		"""

		cases=pd.DataFrame(df_case.groupby(normvars).size()).reset_index()
		ctrls=pd.DataFrame(df_ctrl.groupby(normvars).size()).reset_index()

		if isinstance(normvars, str):
			normvars=[normvars]

		

		ctrls.columns=normvars+['ctrl_recs']
		cases.columns=normvars+['case_recs']

		ctrl_case=pd.merge(cases,ctrls,on=normvars,how='inner')
		ctrl_case['ratio']=(ctrl_case['ctrl_recs']/ctrl_case['case_recs'])
		
		return ctrl_case


	def varnorm(self,df,normvars,depvar='AD',max_mult=None,delete_df=False):

		"""rebalances dataframe to be equal across case and control as defined by depvar=1/0 across a list of variables which must be present in the data
		#df1=df.copy()
		"""
		df_case,df_ctrl=self.case_control(df,depvar=depvar)

		if delete_df:
			del df

		ctrl_case=self.ctrl_case_ratios(df_case,df_ctrl,normvars)
		
		if max_mult==None:
			max_mult=ctrl_case['ratio'].min()

		ctrl_case['case_samp']=max_mult

		return ctrl_case ,df_ctrl,df_case#,cases

	def varnorm_sample(self,df,normvars,depvar,max_mult=None):
		#
		ctrl_case,df_ctrl,df_case=self.varnorm(df,normvars=normvars,depvar=depvar,delete_df=False)
		ctrl_case['recs_sample']=(ctrl_case['case_recs']*ctrl_case['case_samp']).apply(lambda x:np.floor(x)).astype(int)
		df_ctrl=pd.merge(df_ctrl,ctrl_case[normvars+['recs_sample']],on=normvars)
		df_ctrl=pd.DataFrame(df_ctrl.groupby(normvars).\
		apply(lambda x: x.sample(x['recs_sample'].iat[0]))).reset_index(drop=True)
		
		df_out=pd.concat([df_ctrl,df_case],axis=0)
		return df_out


	def eids_var_to_dict(self,df,normvars=['age_when_attended_assessment_centre_f21003_0_0']):

		"""
		create a dictionary of a normalisation variable and lists of unique eids associated with that variable
		"""

		
		normvar=''.join(normvars)


		normvar_list=[c for c in list(df[normvar].unique())]
		normvar_eids=[list(df.loc[(df[normvar]==c,'eid')]) for c in list(df[normvar].unique())]
		normvar_eid_dict=dict(zip(normvar_list,normvar_eids))

		return normvar_eid_dict

	def normvar_samplesize_dict(self,df,norm_var):

		"""
		determine sample size for each normvar as a dictionary - applies to ctrl_case
		"""
		"""if len(norm_vars)==1:
									df['a_var']=df[norm_vars[0]].astype(str)
								elif len(norm_vars)==2:
									df['a_var']=df[norm_vars[0]].astype(str)+df[norm_vars[1]].astype(str)
						"""
		

		df['sample_size']=(df['ratio'].min()*df['case_recs']).apply(np.floor)
		out_dict=dict(zip(df[norm_var],df['case_recs'].astype(int)))

		return out_dict


	def get_indices(self,list_in,start_pos=1000,length=800):
	
		#returns new list taking ino account length of existing list
	
		end_pos=start_pos+length
		length_list=len(list_in)
		
		
		if length_list>end_pos:
			#print("yes")
			#end_pos=start_pos+b
			list_out=list_in[start_pos:end_pos]
			start_pos_new=end_pos
			
		else:
			new_len=end_pos-length_list

			list_out1=list_in[start_pos:length_list]
			list_out2=list_in[0:new_len]
			
			list_out=list_out1+list_out2
			start_pos_new=new_len
	
		return list_out,start_pos_new

	def split_mult_files(self,df,depvar="AD",normvars=['age_when_attended_assessment_centre_f21003_0_0'],iterations=50,mult_fact_max=True,
		multfact=1):

		#purpose to take dataframe and create iterative Monte Carlo based datasets which use all the case data and selectively spread
		#across the control space so all controls which are variable normalised get used

		#convert normvars to one sngle string
		normvar=''.join(normvars)

		df1=df.copy()

		#concatenate values for each norm var
		for i in range(len(normvars)):
			if i==0:
				df1[normvar]=df1[normvars[i]].astype(str)
			else:
				df1[normvar]=df1[normvar]+df1[normvars[i]].astype(str)


		ctrl_case,df_ctrl,df_case=self.varnorm(df1,normvar,depvar=depvar,max_mult=None,delete_df=False)

		case_eids=list(df_case['eid'])

		normvar_eid_dict=self.eids_var_to_dict(df=df_ctrl,normvars=normvars)

		normvar_sample_size_dict=self.normvar_samplesize_dict(df=ctrl_case,norm_var=normvar)

		#reset the dictionaries to ensure the elements are present in both
		normvar_eid_dict=\
	dict(zip([a for a in normvar_eid_dict if a in normvar_sample_size_dict],\
	[normvar_eid_dict[a] for a in normvar_eid_dict if a in normvar_sample_size_dict]))

		normvar_sample_size_dict=\
	dict(zip([a for a in normvar_sample_size_dict if a in normvar_eid_dict],\
	[normvar_sample_size_dict[a] for a in normvar_sample_size_dict if a in normvar_eid_dict]))



		mult_fact_max_val=int(min([np.floor(len(normvar_eid_dict[a])/normvar_sample_size_dict[a]) for a in normvar_eid_dict]))

		if mult_fact_max is True:
			multfact=mult_fact_max_val


		#max_mult

		#print(normvar_eid_dict)

		eids_all=[]

		map_dict=dict()
		start_pos_dict=dict()
		
		for i in range(iterations):


			eids_iter=[]

			for a in normvar_eid_dict:

				
				if i==0:
					list_out,start_pos_new=self.get_indices(normvar_eid_dict[a],start_pos=0,
						length=normvar_sample_size_dict[a]*multfact)
					start_pos_dict[a]=start_pos_new
					#eids_normvars.append(list_out)
					
				else:
					list_out,start_pos_new=self.get_indices(normvar_eid_dict[a],start_pos=start_pos_dict[a],
						length=normvar_sample_size_dict[a]*multfact)
					start_pos_dict[a]=start_pos_new

				eids_iter=eids_iter+list_out

			map_dict[i]=eids_iter+case_eids




		return map_dict

