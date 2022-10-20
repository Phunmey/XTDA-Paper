import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")
fmri_stats = fmri.groupby(['timepoint']).describe()

x = fmri_stats.index
medians = fmri_stats[('signal', '50%')]
medians.name = 'signal'
quartiles1 = fmri_stats[('signal', '25%')]
quartiles3 = fmri_stats[('signal', '75%')]

ax = sns.lineplot(x, medians)
ax.fill_between(x, quartiles1, quartiles3, alpha=0.3)
plt.show()