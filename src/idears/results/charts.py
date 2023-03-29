# Creates economic barchart

import matplotlib.pyplot as plt

class charts():

	def __init__(self):	
		self.recs=15
		return None


	def make_econ_bar(self,df,sort_var='mean_shap',err_var=None,recs=None,title='APOE4 Carriers Feature Importance',
		sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
		tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2],shrink=True,figsize=(3,16),out=True):

		# Setup plot size.

		if recs is None:
			recs=self.recs

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
		ax.set_xticks(tick_vals_use)
		ax.set_xticklabels(tick_vals_use)

		# Reformat x-axis tick labels
		ax.xaxis.set_tick_params(labeltop=True,      # Put x-axis labels on top
								labelbottom=False,  # Set no x-axis labels on bottom
								bottom=False,       # Set no ticks on bottom
								labelsize=10,       # Set tick label size
								pad=-1)             # Lower tick labels a bit

		# Reformat y-axis tick labels
		ax.set_yticklabels(df_bar['Attribute'],      # Set labels again
						ha = 'left')              # Set horizontal alignment to left
		ax.yaxis.set_tick_params(pad=300,            # Pad tick labels so they don't go over y-axis
								labelsize=10,       # Set label size
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
		plt.show() 

		return fig,df_bar           # Set background color to white