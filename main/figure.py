import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

roberta_result = pd.read_csv('../output/roberta_all_df_intention.csv')
plt.figure()
sns.set(rc={"figure.figsize":(18, 6)}) 
sns.set_style("whitegrid")
color_palette = {'Female':'#ff7f00', 'Male':'#377eb8','Unspecified':'#bcbcbc'}
BOX_WIDTH = 0.5

ax = sns.boxplot(x="prompt", y="probability", hue="gender",
                data=roberta_result, showfliers=False, width=BOX_WIDTH, palette=color_palette)
sns.despine(offset=10)

# sns.set(rc={'figure.figsize': (30, 6)}, font_scale=1.2)

plt.xticks(rotation=30, ha='right', fontsize=16)
plt.yticks(fontsize=18)
ax.set_ylabel('Probability', fontsize = 20.0)
ax.set_ylim([0, 0.55])
ax.set(xlabel=None)
ax.set_title("RoBERTa - MH", size=20,fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Male', 'Female', 'Unspecified'], loc='upper left',bbox_to_anchor=(0.85, 1.05),fontsize=16)

plt.savefig('../plots/part1_RoBERTa_whitebackground_16.pdf', bbox_inches="tight")