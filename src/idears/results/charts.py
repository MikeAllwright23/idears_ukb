# Creates economic barchart

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy import stats

from models.ml import *
ml=ml_funcs()

class charts():

	def __init__(self):	
		self.recs=15
		#units dictionary for charts
		self.yaxis_units=dict({'igf1_f30770_0_0':'nmol/L','total_bilirubin_f30840_0_0':'umol/L',
			'AST_ALT_ratio':'AST:ALT ratio',
			'creatinine_enzymatic_in_urine_f30510_0_0':'micromole/L',
			'urate_f30880_0_0':'umol/L',
			'neutrophill_count_f30140_0_0':'10^9 cells/Litre','lymphocyte_count_f30120_0_0':'10^9 cells/Litre',
			'neutrophill_lymphocyte_ratio':'Ratio',
		 'creactive_protein_f30710_0_0':'mg/L','ibuprofen':'Percentage taking Ibuprofen',
		 'cholesterol_f30690_0_0':'mmol/l','hdl_cholesterol_f30760_0_0':'mmol/l',
		 'waist_circumference_f48_0_0':'cm','ldl_direct_f30780_0_0':'mmol/L',
		 'Total ICD10 Conditions at baseline':'',
		 'number_of_treatmentsmedications_taken_f137_0_0':'',
		 'hand_grip_strength_left_f46_0_0':'Kg', 'hand_grip_strength_right_f47_0_0':'Kg',
		 'usual_walking_pace_f924_0_0':'Interview speed',
		   'forced_vital_capacity_fvc_f3062_0_0':'Litres',
		   'glycated_haemoglobin_hba1c_f30750_0_0':'nmol/L'})
		
		self.date_run=str(datetime.now().date())
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.path_figures_pd='Users/michaelallwright/Documents/github/ukb/pd/figures/'
		
		return None


	def make_econ_bar(self,df,sort_var='mean_shap',err_var=None,recs=None,title='APOE4 Carriers Feature Importance',
		sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
		tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2],shrink=True,figsize=(3,16),out=True,line=True):

		# Setup plot size.

		if recs is None:
			recs=self.recs

		#determines the max on y axis
		y_max=recs-0.5

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
		df_bar=df_bar.sort_values(by=sort_var,ascending=True)
		
		custom_palette=['midnightblue' if c<0 else '#E3120B' for c in list(df_bar['corr'])]#'#006BA2'


		# Plot data here
		# plot error bar if specified based on error variable name
		if err_var is None:
			ax.barh(df_bar['Attribute'], df_bar[sort_var].round(3), color=custom_palette, zorder=2)#
		else:
			ax.barh(df_bar['Attribute'], df_bar[sort_var].round(3), color=custom_palette, zorder=2,xerr=df_bar[err_var])#

		# Set custom labels for x-axis

		if tick_vals_use is not None:
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
		if line:
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
		plt.show() 

		return fig,df_bar           # Set background color to white
	

	def dumbell_plot(self,df,att_var,var1,var2):
		# Setup plot size.
		fig, ax = plt.subplots(figsize=(6,9))

		# Create grid 
		# Zorder tells it which layer to put it on. We are setting this to 1 and our data to 2 so the grid is behind the data.
		ax.grid(which="major", axis='both', color='#758D99', alpha=0.6, zorder=1)

		# Remove splines. Can be done one at a time or can slice with a list.
		ax.spines[['top','right','bottom']].set_visible(False)

		# Setup data
		gdp_dumbbell = df.copy()

		# Plot data
		# Plot horizontal lines first
		ax.hlines(y=df[att_var], xmin=df[[var1,var2]].min(axis=1), xmax=df[[var1,var2]].max(axis=1), color='#758D99', zorder=2, linewidth=2, label='_nolegend_', alpha=.8)
		# Plot bubbles next
		ax.scatter(df[var1], df[att_var], label=var1, s=60, color='#006BA2', zorder=3)
		ax.scatter(df[var2], df[att_var], label=var2, s=60, color='#DB444B', zorder=3)

		# Set xlim
		x_min=df[[var1,var2]].min(axis=1).min()-0.1
		x_max=df[[var1,var2]].max(axis=1).max()+0.1

		print(x_min,x_max)
		ax.set_xlim(x_min, x_max)

		# Reformat x-axis tick labels
		ax.xaxis.set_tick_params(labeltop=True,      # Put x-axis labels on top
								labelbottom=False,  # Set no x-axis labels on bottom
								bottom=False,       # Set no ticks on bottom
								labelsize=15,       # Set tick label size
								pad=-1)             # Lower tick labels a bit

		# Reformat y-axis tick labels
		ax.set_yticklabels(gdp_dumbbell[att_var],       # Set labels again
						ha = 'left')              # Set horizontal alignment to left
		ax.yaxis.set_tick_params(pad=300,            # Pad tick labels so they don't go over y-axis
								labelsize=15,       # Set label size
								bottom=False)       # Set no ticks on bottom/left

		# Set Legend
		ax.legend([var1,var2], loc=(-0.9,1.09), ncol=2, frameon=False, handletextpad=-.1, handleheight=1)

		# Add in line and tag
		ax.plot([-0.59	, .9],                 # Set width of line
				[1.17, 1.17],                # Set height of line
				transform=fig.transFigure,   # Set location relative to plot
				clip_on=False, 
				color='#E3120B', 
				linewidth=.6)
		ax.add_patch(plt.Rectangle((-0.59,1.17),               # Set location of rectangle by lower left corder
								0.05,                       # Width of rectangle
								-0.025,                      # Height of rectangle. Negative so it goes down.
								facecolor='#E3120B', 
								transform=fig.transFigure, 
								clip_on=False, 
								linewidth = 0))

		
		# Add in title and subtitle
		ax.text(x=-0.59, y=1.09, s='SHAP Comparisons', transform=fig.transFigure, ha='left', fontsize=16, weight='bold', alpha=.8)
		ax.text(x=-0.59, y=1.04, s="Difference between groups", transform=fig.transFigure, ha='left', fontsize=14, alpha=.8)

		# Set source text
		ax.text(x=-0.59, y=0.04, s="""Source: "UKB""", transform=fig.transFigure, ha='left', fontsize=9, alpha=.7)


		return fig,ax
	

	def analysis_boxplots(self,df,dis_date='parkins_date',disease='PD',vars=['igf1_f30770_0_0'],
	splitvar='sex_f31_0_0',agemin=50,agemax=70,labels=dict({1:'Female',0:'Male'}),varnames='test',exc_deaths=False,dis_label=True,
	agenormvars=[],agevar='age_when_attended_assessment_centre_f21003_0_0',min_dis_bef=-20,max_dis_aft=15):

		# copy df
		df=df.copy()


		compgroups=list(['No '+disease,'-10>-5','-5>0','0>5'])

		# keep key variables and specific set filtered for
		cols_use=['eid','years_'+disease,disease,splitvar,agevar]+vars

		#filter the period in disease trajectory

		mask=((df['years_'+disease]<=max_dis_aft)&(df['years_'+disease]>=min_dis_bef))|pd.isnull(df['years_'+disease])
		df=df.loc[mask,cols_use]

		def get_dis_stage(x,disease='PD'):
			if x<-10:
				y='<10'
			elif x>=-10 and x<=-5:
				y='-10>-5'
			elif x>-5 and x<=0:
				y='-5>0'
			elif x>0 and x<=5:
				y='0>5'
			elif x>5:
				y='5+'
			else:
				y='No '+disease
			return y
		
		def ttest(df,bdown,var,disease='PD'):
			ttest_vals=stats.ttest_ind(df[(df['dis_stage']==bdown)][var], 
					df[(df['dis_stage']=='No '+disease)][var])
			
			return ttest_vals
		
		def getpval_arr(pval_array,length=3):

			pvals_out=[]
			for k in list(np.arange(length)):
				sig='ns'

				if pval_array[k]<0.001:
					sig='***'
				elif pval_array[k]<0.01:
					sig='**'
				elif pval_array[k]<0.05:
					sig='*'
				pvals_out.append(sig)

			return pvals_out
		
		def agenorm(df,var,normvar='age_when_attended_assessment_centre_f21003_0_0'):

			df2=df.loc[pd.notnull(df[var])&(df[var]!=np.inf)]
			df_sum=pd.DataFrame(df2.groupby([normvar]).agg({var:['mean']})).reset_index()
			df_sum.columns=[normvar,'mean'+var]

			df=pd.merge(df,df_sum,on=normvar,how='left')

			df[var]=df[var].mean()*df[var]/df['mean'+var]
			df.drop(columns=['mean'+var],inplace=True)
			return df
		
		df['dis_stage']=df.apply(lambda x:get_dis_stage(disease=disease,x=x['years_'+disease]),axis=1)

		
		compgroups=[c for c in compgroups if c in list(df['dis_stage'].unique())]

		for a in agenormvars:
			df=agenorm(df,a)

		k=len(vars)

		fig = plt.figure(figsize=(15*k, 10*k))
		grid = plt.GridSpec(k, k, hspace=0.45, wspace=0.3)

		ttestvals,lq_vals,med_vals,uq_vals,pvallist,varnameslist,splitnames,comp_groups,std_vals,grpsize_arr=\
	[],[],[],[],[],[],[],[],[],[]

		#unique splitvars
		splitvars=list(set(list(df.loc[pd.notnull(df[splitvar]),splitvar].unique())))


		for i,v in enumerate(vars):
			for j,t in enumerate(splitvars):

				#filter data to exclude infinite values for each var, null values and only when splitvar is relevant one
				mask=(pd.notnull(df[v]))&(df[v]!=np.inf)&(df[v]!=np.nan)&(df[splitvar]==t)
				df_use=df.loc[mask,]

				

				#sort by disease stage to be in correct order
				df_use.sort_values(by='dis_stage',inplace=True)
				
				#initiate subplot
				ax=fig.add_subplot(grid[i, j])
				ax=sns.boxplot(x=df_use['dis_stage'],y=df_use[v],order=compgroups,showfliers = False,color='skyblue')
				plt.xticks(fontsize='35')
				plt.yticks(fontsize='35')

				#plt.ylabel(v, fontsize=24)
				if v in self.yaxis_units:
					unit=self.yaxis_units[v]
				else:
					unit='%'
				plt.ylabel(unit,fontsize='30')
				plt.xlabel(labels[t]+'s: '+str(ml.mapvar(v)), fontsize='35')
				plt.xlabel('Years of '+disease)
				plt.title(labels[t]+'s: '+str(ml.mapvar(v)), fontsize='35')

				#whisker locations - find top within group whisker
				
				avg=df_use[v].mean() #mean val
				max_val=df_use[v].max() #max val
				min_val=df_use[v].min() #min val
				q_25=df_use[v].quantile(0.25) #lower quartile
				q_75=df_use[v].quantile(0.75) #upper quartile
				iqr=q_75-q_25 #iq range
			   
				iqr_pos = q_75+1.5*iqr if (q_75+1.5*iqr)<max_val else max_val
				iqr_neg = q_25-1.5*iqr if (q_25-1.5*iqr)>min_val else min_val

				iqr_pos_arr,iqr_neg_arr,pvallist_small=[],[],[]
				
				for q,m in enumerate(compgroups):

					ttest_vals=ttest(df_use,m,v,disease=disease)
					mask_dis_stage=(df_use['dis_stage']==m)

					df_use_ds=df_use.loc[mask_dis_stage,]

					max_val=df_use_ds[v].max()
					min_val=df_use_ds[v].min()
					q_25=df_use_ds[v].quantile(0.25)
					q_75=df_use_ds[v].quantile(0.75)
					med=df_use_ds[v].quantile(0.5)
					grpsize=df_use_ds.shape[0]
					iqr=q_75-q_25

					std_val=round(df_use_ds[v].std(),3)

					iqr_pos = q_75+1.5*iqr if (q_75+1.5*iqr)<max_val else max_val
					iqr_pos_arr.append(iqr_pos)
					iqr_neg = q_25-1.5*iqr if (q_25-1.5*iqr)>min_val else min_val
					iqr_neg_arr.append(iqr_neg)

				   
					ttest_val_inc=round(df_use_ds[v].mean(),3)
					pval_inc=round(ttest_vals[1],6)

					grpsize_arr.append(grpsize)
					ttestvals.append(ttest_val_inc)
					med_vals.append(med)
					lq_vals.append(q_25)
					uq_vals.append(q_75)


					pvallist.append(pval_inc)
					pvallist_small.append(pval_inc)
					varnameslist.append(v)
					splitnames.append(t)
					comp_groups.append(m)
					std_vals.append(std_val)

				#print(pvallist_small)
				pvals_signs=getpval_arr(pval_array=pvallist_small,length=len(compgroups))

				#for k in list(np.arange(len(compgroups))+1):

				
				for k in [1,2,3]:

				
					sig=pvals_signs[k]

					if sig !='ns':
						x1, x2 = 0, k  
						y, h, col = iqr_pos + (iqr_pos_arr[k]-iqr_neg_arr[k])/5, k*(iqr_pos_arr[k]-iqr_neg_arr[k])/5, 'black'
						plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
						#plt.text((x1+x2)*.5, y+h, 'p= '+str(pvallist_small[k])+' '+str(sig), ha='center', va='bottom', color=col,
						#	fontsize='24')
						plt.text((x1+x2)*.5, y+h*0.95, str(sig), ha='center', va='bottom', color=col,
							fontsize='24')




		plt.savefig(self.path_figures_pd+self.date_run+"_"+varnames+'.jpg', dpi=300,bbox_inches='tight')
		plt.show()
		
		pvals_df=pd.DataFrame({'Variable':varnameslist,'Split':splitnames,'Group Size':grpsize_arr,'Years into disease':comp_groups,
			'Mean':ttestvals,'Median':med_vals,'lower quartile':lq_vals,
			'upper quartile':uq_vals,'p value':pvallist,'standard deviations':std_vals})
		
		#dictionary of outputs - original df and p value df
		outs=dict({'df':df,'pvals_df':pvals_df})

		return outs