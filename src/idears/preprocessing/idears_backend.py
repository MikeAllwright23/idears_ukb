
"""
Module to 

1. Create trainign data for each disease, age breakdown etc.
2. Run logistic regression and take odds ratios of feature sets
3. Validate and return AUCs, precision scores etc.

"""


path='../../../data/ukb/ad/'
import sys
sys.path.append('/Users/michaelallwright/Documents/github/ukb/pipeline/')
from data_gen import *
from data_proc import *
from ml import *

import json
apoe_set=["Non Carriers","Carriers","All"]
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
di=data_import()
dp=data_proc()
ml=ml_funcs()

class idears():

	def __init__(self):	


		self.dis_icd10_dict=\
		{'Diabetes':['E10','E11'],
		 'MND': ['G122'],
		 'FTD': ['G310'],
		 'VD': ['F01'],
		 'AD': ['G30'],
		 'PD': ['G20'],
		 'Other': ['G31'],
		 'All': ['G20', 'G30', 'G31']}

		self.dis_exc_vars_dict={'Diabetes':'diabetes'}

		self.age_dict={'55-70': [55,70],
		 '60-70': [60,70],
		 '50-60': [50,60]}

		self.gend_dict={'Male': [1],
		 'Female': [0],
			'All':[0,1]}

		self.gend_dict_extcols={'Male': ['age_when_attended_assessment_centre_f21003_0_0'],
		 'Female': ['age_when_attended_assessment_centre_f21003_0_0'],
			'All':['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0']}

		# raw data across the board
		self.df_mod=pd.read_parquet(di.path+"ukb_df_processed2022-11-15.parquet")

		#field lookup table
		self.df_fields=di.get_field_names()

	def get_mod_fields():
		mask=(self.df_fields['Modifiable']==1)
		return df_fields.loc[mask,]

	def find_cols_filt(df):
		all_col_names=list(self.df_fields['col.name'])
		cols=[]
		for c in all_col_names:
			for d in df.columns:
				if str(c) in str(d):
					cols.append(c)
		mask=df_fields['col.name'].isin(cols)
		df_fields_filt=self.df_fields.loc[mask,].copy()
		df_fields_filt['field_type1']=df_fields_filt['field_type1'].astype(str).apply(lambda x:x.strip())     
					
		return df_fields_filt

	def df_cols(df,cols):
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
	    
    

	def filt(df,dis,fields_include,extcols):
		df_fields_filt=find_cols_filt(df,self.df_fields)

		mask=df_fields_filt['field_type1'].isin(fields_include)
		# ensure max 1 of each
		cols_use=list(set(list(df_fields_filt.loc[mask,'col.name'])+['eid']+extcols+[dis]))
		cols_use=df_cols(df,cols_use)
		return df[cols_use]

	def split_bdown(self,df,bvar='breakdown'):
		df['disease']=df[bvar].apply(lambda x:x.split('|')[0])
		df['Age Range']=df[bvar].apply(lambda x:x.split('|')[1])
		df['Gender']=df[bvar].apply(lambda x:x.split('|')[2])
		return df

	def determine_cols(self,df,dis,fields_include_use,age,gen):
			#filter data to the age ranges and genders associated with the dictionary given for each iteration
		mask=(df['age_when_attended_assessment_centre_f21003_0_0'].between(self.age_dict[age][0],self.age_dict[age][1]))&\
		(df['sex_f31_0_0'].isin(self.gend_dict[gen]))
		df3=df.loc[mask,]
		bdown='|'.join([dis,age,gen])
		print(bdown,str(df3[dis].sum()),' participants')

		if fields_include_use==["All"]:
			#simple logic when all features selected
			df3=ml.rename_cols(df3)
		else:
			if fields_include_use==["Modifiable"]:
			#modifiable features use a different column to identify
				cols=[c for c in df3.columns if c=='eid' or c==dis or c in extcols or\
				c in list(df_fields_mod['col.name'])]
				df3=df3[cols]
				df3=ml.rename_cols(df3)
			
			else:
				#otherwise the fields used are the ones specified
				df3=filt(df=df3,df_fields=df_fields,fields_include=fields_include_use,dis=dis,extcols=extcols)
				df3=ml.rename_cols(df3)  
		#pipe joined lookup variable for selections
		
		
		
		
		#split train and test data for this filtered dataframe
		df_tr,df_te=ml.train_test(df=df3,depvar=dis,test_size=0.3,random_state=421)

		return df3,bdown,df_tr,df_te 

	def create_train_test(self,extcols,name,fields_include_use,dis_icd10_dict,age_dict,gend_dict):
		
		#set up an empty dictionary to store a list of train and test dataframe within each category
		df_dict=dict()

		#initiate df
		df=self.df_mod.copy()
		
		for dis in self.dis_icd10_dict:
			#loop through the disease name to ICD10 list mapping
			#create a model dataset for each disease 
			
			df2=di.create_model_data(df=df,depvar=dis,icd10s=dis_icd10_dict[dis],infile="ukb_df_processed2022-11-15.parquet")
			print(df2.shape)
			
			for agex in self.age_dict:
				#loop through the age ranges specified
				
				for genx in self.gend_dict:
					#loop through the genders specified

					df3,bdown,df_tr,df_te=self.determine_cols(fields_include_use=fields_include_use,dis=dis,df=df2,age=agex,gen=genx)
   
					#store train and test data in the dictionary under the lookup of bdown
					df_dict[bdown]=[df_tr,df_te]
						
		return df_dict

		


	def get_aucs_all(self,df_dict,dis_icd10_dict,age_dict,gend_dict,gend_dict_extcols,dis_exc_vars_dict,iters=10):
		
		aucs=[]
		bdowns=[]
		feats_full=pd.DataFrame([])


		for dis in self.dis_icd10_dict: 
			for agex in self.age_dict:
				for genx in self.gend_dict:
						
					bdown='|'.join([dis,agex,genx])
		
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
					df_dict1_a=dp.split_mult_files(df=df_train1,depvar=dis,\
					normvars=gend_dict_extcols[genx],iterations=iters,mult_fact_max=1)
					
					#split normalised datasets for test data
					df_dict1_a_test=dp.split_mult_files(df=df_test1,depvar=dis,\
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

		return df_auc,feats_full

		def get_avg_vals(df_dict,dis_icd10_dict,age_dict,gend_dict):
		
			bdowns=[]
			cols=[]
			case_means=[]
			ctrl_means=[]
			pvals=[]
			
			for dis in self.dis_icd10_dict: 
				for agex in self.age_dict:
					for genx in self.gend_dict:
						
						bdown='|'.join([dis,agex,genx])
			
						print(bdown)

						df_train1,df_test1=df_dict[bdown]
						#df1=pd.concat([df_train1,df_test1])
						df1=df_test1.copy()
						
						remwords='|'.join([c for c in ml.remwords if c!='eid'])+'|'+'recent_feelings_of_tiredness_or_low'

						remcols=[c for c in df1.columns if re.search(remwords,c.lower())
								 or ('obj' in str(df1[c].dtype)) or c==dis]
						
						cols_compare=[c for c in df1.columns if c not in remcols ]
						
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
			
			df_avg_vals=split_bdown(df_avg_vals,bvar='breakdown')
			
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

		def get_pop_chars(df_dict,dis_icd10_dict,age_dict,gend_dict):
		
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
			
			df_out=split_bdown(df_out)
			
			return df_out