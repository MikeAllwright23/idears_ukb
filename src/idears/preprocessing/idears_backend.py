"""
Module to 
1. Create trainign data for each disease, age breakdown etc.
2. Run logistic regression and take odds ratios of feature sets
3. Validate and return AUCs, precision scores etc.
"""
import sys
sys.path.append('../')
from scipy import stats

from preprocessing.data_proc import *
from models.mlv2 import *

dp=data_proc()
nm=normalisations()
ml=ml_funcs()

class Idears():
	def __init__(self):
		
		self.path='../data/'
		self.field_names=self.path+'metadata/ukb_field_names.xlsx'
		self.dis_icd10_dict=\
		{#'Diabetes':['E10','E11'],
		 'MND': ['G122'],
		 'FTD': ['G310'],
		 'VD': ['F01'],
		 'AD': ['G30'],
		 'PD': ['G20'],
		 'Other': ['G31'],
		 'All': ['G20', 'G30', 'G31']}
		
		self.dis_exc_vars_dict={'Diabetes':'diabetes'}

		self.age_dict={'50-70': [50,70],'55-70': [55,70],
		 '60-70': [60,70],
		 '50-60': [50,60]}
		
		self.apoe4_dict={'Carriers': [1],'Non Carriers': [0],'All':[0,1]}

		self.gend_dict={'Male': [1],
		 'Female': [0],
			'All':[0,1]}

		self.gend_dict_extcols={'Male': ['age_when_attended_assessment_centre_f21003_0_0'],
		 'Female': ['age_when_attended_assessment_centre_f21003_0_0'],
			'All':['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0']}

		# raw data across the board
		#self.df_mod=pd.read_parquet(self.path+"ukb_df_processed2022-11-15.parquet")

		#field lookup table
		self.df_fields=pd.read_excel(self.field_names,sheet_name='fieldnames_full')

		return None


	def get_mod_fields(self):
		mask=(self.df_fields['Modifiable']==1)
		return self.df_fields.loc[mask,]


	def rename_cols(self,df):
		#ensures cols can be modelled
		df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
		return df


	def find_cols_filt(self,df):
		all_col_names=list(self.df_fields['col.name'])
		cols=[]
		for c in all_col_names:
			for d in df.columns:
				if str(c) in str(d):
					cols.append(c)
		mask=self.df_fields['col.name'].isin(cols)
		df_fields_filt=self.df_fields.loc[mask,].copy()
		df_fields_filt['field_type1']=df_fields_filt['field_type1'].astype(str).apply(lambda x:x.strip())     				
		return df_fields_filt


	def df_cols(self,df,cols):
		cols_out=[]
		for c in cols:
			for d in df.columns:
				if c in d:
					cols_out.append(d)
		return cols_out


	def get_col_incidences(self,df,dis,col):
		#split col into high and low
		mask=pd.notnull(df[col])
		df1=df.loc[mask,]
		df1[col+'q']=pd.qcut(df1[col],q=2,labels=False).astype(int)
		return dict(df1.groupby(col+'q')[dis].mean())


	def filt(self,df,dis,fields_include,extcols):
		df_fields_filt=self.find_cols_filt(df,self.df_fields)

		mask=df_fields_filt['field_type1'].isin(fields_include)
		# ensure max 1 of each
		cols_use=list(set(list(df_fields_filt.loc[mask,'col.name'])+['eid']+extcols+[dis]))
		cols_use=self.df_cols(df,cols_use)
		return df[cols_use]


	def split_bdown(self,df,bvar='breakdown'):
		df['disease']=df[bvar].apply(lambda x:x.split('|')[0])
		df['Age Range']=df[bvar].apply(lambda x:x.split('|')[1])
		df['Gender']=df[bvar].apply(lambda x:x.split('|')[2])
		return df


	def determine_cols(self,df,dis,fields_include_use,age,gen,apoe4):
			#filter data to the age ranges and genders associated with the dictionary given for each iteration
		mask=(df['age_when_attended_assessment_centre_f21003_0_0'].between(self.age_dict[age][0],self.age_dict[age][1]))&\
		(df['sex_f31_0_0'].isin(self.gend_dict[gen]))&(df['APOE4_Carriers'].isin(self.apoe4_dict[apoe4]))
		df3=df.loc[mask,]
		bdown='|'.join([dis,age,gen,apoe4])
		print(bdown,str(df3[dis].sum()),' participants')

		if fields_include_use==["All"]:
			#simple logic when all features selected
			df3=self.rename_cols(df3)
		else:
			if fields_include_use==["Modifiable"]:
			#modifiable features use a different column to identify
				mask=self.df_fields['Modifiable']==1
				df_fields_mod=self.df_fields.loc[mask,]
				
				cols=[c for c in df3.columns if c=='eid' or c==dis or c in self.gend_dict_extcols[gen] or\
				c in list(self.df_fields_mod['col.name'])]
				df3=df3[cols]
				df3=self.rename_cols(df3)
			
			else:
				#otherwise the fields used are the ones specified
				df3=self.filt(df=df3,df_fields=self.df_fields,fields_include=fields_include_use,dis=dis,extcols=self.gend_dict_extcols[gen])
				df3=self.rename_cols(df3)  
		#pipe joined lookup variable for selections
		
		
		#split train and test data for this filtered dataframe
		df_tr,df_te=self.train_test(df=df3,depvar=dis,test_size=0.3,random_state=421)

		return df3,bdown,df_tr,df_te 


	def train_test(self,df,depvar='AD',test_size=0.3,random_state=42):
		# splits in random state train and test for given depvar

		mask=(df[depvar]==1)
		cases=df.loc[mask,]
		ctrls=df.loc[~mask,]
		test_case=cases.sample(frac=test_size,random_state=random_state)
		test_ctrl=ctrls.sample(frac=test_size,random_state=random_state)
		df_test=pd.concat([test_case,test_ctrl],axis=0)
		mask=~(df['eid'].isin(df_test['eid']))
		df_train=df.loc[mask,]
		return df_train,df_test


	def create_train_test(self,fields_include_use,df=None,ages=None,gends=None,diseases=None,apoe4s=None):
		
		#set up an empty dictionary to store a list of train and test dataframe within each category
		df_dict=dict()

		#initiate df
		#if df is None:
			#df=self.df_mod.copy()

		if ages is None:
			ages=self.age_dict
		if gends is None:
			gends=self.gend_dict
		if apoe4s is None:
			apoe4s=self.apoe4_dict
		if diseases is None:
			diseases=self.dis_icd10_dict
	
		for dis in diseases:
			#loop through the disease name to ICD10 list mapping
			#create a model dataset for each disease 
			
			#if df is None:
			df=dp.create_model_data(depvar=dis,icd10s=diseases[dis],infile="ukb_df_processed2022-11-15.parquet")
			print(df[dis].sum())
			
			
			for agex in ages:
				#loop through the age ranges specified
				
				for genx in gends:
					#loop through the apoe4s specified
					for apoe4 in apoe4s:

						df3,bdown,df_tr,df_te=self.determine_cols(fields_include_use=fields_include_use,dis=dis,df=df,age=agex,gen=genx,apoe4=apoe4)
	
						#store train and test data in the dictionary under the lookup of bdown
						df_dict[bdown]=[df_tr,df_te]
							
		return df_dict


	def get_aucs_all(self,df_dict,gend_dict_extcols,dis_exc_vars_dict,iters=10,ages=None,gends=None,apoe4s=None,diseases=None):
		
		aucs=[]
		bdowns=[]
		feats_full=pd.DataFrame([])

		if ages is None:
			ages=self.age_dict
		if gends is None:
			gends=self.gend_dict
		if apoe4s is None:
			apoe4s=self.apoe4_dict
		if diseases is None:
			diseases=self.dis_icd10_dict

		for dis in diseases: 
			for agex in ages:
				for genx in gends:
					#loop through the apoe4s specified
					for apoe4 in apoe4s:
						
						bdown='|'.join([dis,agex,genx,apoe4])
			
						print(bdown)

						df_train1,df_test1=df_dict[bdown]
						
						#specific words to remove for that disease if it is in the dictionary
						dis_rems=''
						if dis in dis_exc_vars_dict:
							dis_rems=dis_exc_vars_dict[dis]
						
						#combine all words which, if they are contained in a column that column will not be taken to model
						remwords='|'.join([c for c in ml.remwords if c!='eid'])+\
						'|'+'recent_feelings_of_tiredness_or_low'+dis_rems
						
						#cols associated with remove words remove all date columns too
						remcols=[c for c in df_train1.columns if re.search(remwords,c.lower())
								or ('obj' in str(df_train1[c].dtype) and c!='eid' and c!=dis) or ('_date_' in c)]
						
						#actual cols to model
						keepcols=[c for c in df_train1.columns if c not in remcols] 
						df_train1=df_train1[keepcols]
						df_test1=df_test1[keepcols]
						
						#split normalised datasets for training data
						df_dict1_a=nm.split_mult_files(df=df_train1,depvar=dis,\
						normvars=gend_dict_extcols[genx],iterations=iters,mult_fact_max=1)
						
						#split normalised datasets for test data
						df_dict1_a_test=nm.split_mult_files(df=df_test1,depvar=dis,\
						normvars=gend_dict_extcols[genx],iterations=iters,mult_fact_max=1)


						for j in range(len(df_dict1_a)):

							mask=(df_train1['eid'].isin(df_dict1_a[j]))
							df_train2a=df_train1.loc[mask,]
							df_train2a.drop(columns=gend_dict_extcols[genx],inplace=True)

							feats_all=ml.get_shap_feats(df_train2a.drop(columns='eid'),depvar=dis)
							feats_all['iteration']=j
							feats_all['breakdown']=bdown
							feats_full=pd.concat([feats_full,feats_all],axis=0)

							mask=(df_test1['eid'].isin(df_dict1_a_test[j]))
							df_test2a=df_test1.loc[mask,]
							df_test2a.drop(columns=gend_dict_extcols[genx],inplace=True)

							X_train,y_train=ml.make_xy(df_train2a,depvar=dis)
							X_test,y_test=ml.make_xy(df_test2a,depvar=dis)

							model=ml.fit_model(X_train,y_train)
							score=ml.auc_score(valid_x=X_test,valid_y=y_test,model=model,mod_name='XGB')
							aucs.append(score)
							bdowns.append(bdown)

		df_auc=pd.DataFrame({'breakdown':bdowns,'auc':aucs})
					#sns.boxplot(df_auc,y='disease',x='auc',color='grey')
			
		df_auc=self.split_bdown(df_auc,bvar='breakdown')
		feats_full=self.split_bdown(feats_full,bvar='breakdown')
		df_feats_sum=self.get_shap_sum(feats_full)

		return df_auc,df_feats_sum


	def field_map(self):
		""" Map the UKB fields and own remapping of variables so the correct variable
		names show.
		"""
		map_fields=dict(zip(self.df_fields['col.name'],self.df_fields['Field']))
		map_fields=dict(zip(list(self.df_fields['col.name']),[ml.variablemap[c] if c in ml.variablemap else map_fields[c] for c in list(self.df_fields['col.name'])]))
		return map_fields


	def get_shap_sum(self,df):
		""" Summarise the SHAP charts to take mean and average correlation,
		remap the field nmes ready for display
		"""

		map_fields=self.field_map()

		df_s=pd.DataFrame(df.groupby(['breakdown','Attribute']).agg({'mean_shap':['sum','count','std'],'corr':'mean'})).reset_index()
		df_s.columns=[c[0]+c[1] for c in df_s.columns]
		df_s['mean_shap']=df_s['mean_shapsum']/df_s.groupby(['breakdown','Attribute'])['mean_shapsum'].transform('count')


		df_s['corr']=df_s['corrmean']

		#copy original for later use
		df_s['Attribute_original']=df_s['Attribute'].copy()
		df_s['Attribute_new']=df_s['Attribute'].map(map_fields)#.astype(str)

		# relabel only if relabel in the map
		mask=pd.notnull(df_s['Attribute_new'])
		df_s.loc[mask,'Attribute']=df_s.loc[mask,'Attribute_new']

		mask=~pd.isnull(df_s['Attribute'])
		df_s=df_s.loc[mask,]
		return df_s


	def get_avg_vals(self,df_dict,cols_compare,diseases=None,ages=None,gends=None,apoe4s=None):

		""" Cycle through the iterations to return the mean values alongside associated
		p values for a set of variables determined through SHAP
		"""
	
		bdowns=[]
		cols=[]
		case_means=[]
		ctrl_means=[]
		pvals=[]

		if ages is None:
			ages=self.age_dict
		if gends is None:
			gends=self.gend_dict
		if apoe4s is None:
			apoe4s=self.apoe4_dict
		if diseases is None:
			diseases=self.dis_icd10_dict
		
		for dis in diseases: 
			for agex in ages:
				for genx in gends:
					for apoe in apoe4s:
					
						bdown='|'.join([dis,agex,genx,apoe])
			
						print(bdown)

						df_train1,df_test1=df_dict[bdown]
						#df1=pd.concat([df_train1,df_test1])
						df1=df_test1.copy()
						
						#comparisons to look at
						cols_compare=[c for c in df1.columns if c in cols_compare]
						
						for k in cols_compare:
							mask=(df1[dis]==1)&pd.notnull(df1[k])
							mask1=(df1[dis]==0)&pd.notnull(df1[k])
							case_val=df1.loc[mask,k].mean()
							ctrl_val=df1.loc[mask1,k].mean()
							
							pval=stats.ttest_ind(df1.loc[mask,k], df1.loc[mask1,k])[1]
							
							bdowns.append(bdown)
							cols.append(k)
							case_means.append(case_val)
							ctrl_means.append(ctrl_val)
							pvals.append(pval)
							
		df_avg_vals=pd.DataFrame({'breakdown':bdowns,'Attribute':cols,'case_mean':case_means,
									'ctrl_mean':ctrl_means,'p value':pvals})
		
		df_avg_vals=self.split_bdown(df_avg_vals,bvar='breakdown')
		
		return df_avg_vals


	def pop_char1(self,df_dict,dis,age,gen):

		bdown='|'.join([dis,agex,genx])
			
		print(bdown)

		df_train1,df_test1=df_dict[bdown]
		df1=pd.concat([df_train1,df_test1])
		#df1=df_test1.copy()
		
		mask=(df1[dis]==1)
		
		age_mean=round(df1['age_when_attended_assessment_centre_f21003_0_0'].mean(),2)
		age_std=round(df1['age_when_attended_assessment_centre_f21003_0_0'].std(),2)
		
		casenum=df1.loc[mask,].shape[0]
		ctrlnum=df1.loc[~mask,].shape[0]

		return bdown,age_mean,age_std,casenum,ctrlnum


	def get_pop_chars(self,df_dict,dis_icd10_dict,age_dict,gend_dict):
		"""
		Compute population characteristics for all combinations of disease, age group, and gender.

		Args:
			df_dict (dict): A dictionary containing data frames for each combination of disease, age, and gender.
			dis_icd10_dict (dict): A dictionary containing ICD-10 codes for each disease.
			age_dict (dict): A dictionary containing age groups.
			gend_dict (dict): A dictionary containing genders.

		Returns:
			A data frame containing the following columns:
			- breakdown (str): A string representation of the breakdown (e.g., "A00|0-4|Male").
			- mean_age (float): The mean age of the population.
			- std_age (float): The standard deviation of the age of the population.
			- cases (int): The number of cases in the population.
			- controls (int): The number of controls in the population.
		"""
		bdowns=[]
		age_means=[]
		age_stds=[]
		cases=[]
		controls=[]
		
		for dis in self.dis_icd10_dict: 
			for agex in self.age_dict:
				for genx in self.gend_dict:

					bdown,age_mean,age_std,casenum,ctrlnum=self.pop_char1(df_dict,dis=dis,age=agex,gen=genx)

					bdowns.append(bdown)
					age_means.append(age_mean)
					age_stds.append(age_std)
					cases.append(casenum)
					controls.append(ctrlnum)
						
		df_out=pd.DataFrame({'breakdown':bdowns,'mean_age':age_means,'std_age':age_stds,'cases':cases,'controls':controls})
		df_out=self.split_bdown(df_out)
		
		return df_out
	

if __name__=='__main__':

	"""
	Generate all key data so we can run and output to a given location
	"""
	run_shap=False

	ib=Idears()
	print("start")
	df_dict=ib.create_train_test(fields_include_use=["All"],diseases=None,df=None,ages=None,gends=None,apoe4s=None)#{'AD':['G30']}
	
	if run_shap:
		print("dict done")
		df_auc,df_feats_sum=ib.get_aucs_all(df_dict,ib.gend_dict_extcols,ib.dis_exc_vars_dict,
					diseases=None,iters=2,ages=None,gends=None,apoe4s=None)#{'AD':['G30']}
		
		#output the data locally for input to streamlit app
		df_auc.to_csv(ib.path+'df_auc.csv')
		df_feats_sum.to_csv(ib.path+'df_feats_sum.csv')

	df_feats_sum=pd.read_csv(ib.path+'df_feats_sum.csv')
	cols_compare=list(df_feats_sum.sort_values(by='mean_shap',ascending=False)['Attribute_original'])
	#print(cols_compare)
	df_avg_vals=ib.get_avg_vals(df_dict,cols_compare,diseases=None)#{'AD':['G30']})
	df_avg_vals.to_csv(ib.path+'df_avg_vals.csv')
   