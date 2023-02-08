
#%%
#!%matplotlib widget
import pandas
import matplotlib as mpl
import matplotlib.pyplot
import numpy as np
import scipy.stats

from param_pca import param_pca

#%%
# Load expression data
gtex_data = pandas.read_csv(
    "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/gene_tpm/gene_tpm_2017-06-05_v8_liver.gct.gz",
     sep="\t",
     skiprows=[0,1],
     index_col=[0,1,2],
)

#%%
# Load subject metadata, including age
gtex_subj_metadata = pandas.read_csv(
    "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
    sep="\t",
    index_col=0,
)
gtex_subj_metadata['sex'] = gtex_subj_metadata['SEX'].map({1: "Male", 2:"Female"})

#%%
# Extract subject IDs from the column names
import re
samp_id_re = re.compile(r"(GTEX-[0-9A-Z]+)-")
subj_id = pandas.Series(
    gtex_data.columns.map(lambda x: samp_id_re.match(x).groups()[0]),
    index = gtex_data.columns
)

# %%
# Match each sample to its age
subj_age_cat = subj_id.map(gtex_subj_metadata.AGE)
# Map each of the age ranges to its midpoint
def age_cat_midpoint(cat):
    low = int(cat[:2])
    high = int(cat[3:])+1
    return (low+high)//2
subj_age = subj_age_cat.apply(age_cat_midpoint)

# Determine the death type
# The top principal component is separation of ventilator and non-ventilator deaths
subj_death_hardy = subj_id.map(gtex_subj_metadata.DTHHRDY)

# %%
# Fit the ParamPCA model
metadata = pandas.DataFrame({
    "age": subj_age,
})
results = param_pca.ParamPCA(
    data = gtex_data.T,
    metadata = metadata,
    formula = "center(age)",
    r = 2,
    R = 50,
    verbose = True,
    learning_rate = 0.001,
    niter = 100,
)

# %%
print(results.summary())

# %%
# Plot the PCA scores of all points for the various marginal PCAs at ages ranging from 30 to 70
# The PCA at age 70 is plotted, with the trajectories of each subject across the different age PCAs
# shown in grey

# The figure shows that the top PC is dominated by death type according to the Hardy scale
# We plot this as Ventilator (circles) vs all others (Xs). We also see that the trajectory taken as
# we alter the PCA projection with the age is nearly exactly moves subjects vertically only.
# This demonstrates that PC 0 (and hence ventilator vs non-ventilator death) is age-independent while the
# second component captures a component of variation in age.

fig, ax = mpl.pyplot.subplots()
cmap = mpl.colormaps["inferno"]
min_age = metadata['age'].min()
max_age = metadata['age'].max()
colors = 2*(results.metadata['age'] - min_age) / (min_age + max_age)
scores =  np.array([results.PCA_scores_at({"age": age}) for age in range(30,71,1)])
is_ventilator = subj_death_hardy == 0.0 # 0 is ventilator
h1 = ax.scatter(scores[-1, is_ventilator, 0], scores[-1, is_ventilator, 1], c=cmap(colors[is_ventilator]), s=50, marker="o")
h2 = ax.scatter(scores[-1, ~is_ventilator, 0], scores[-1, ~is_ventilator, 1], c=cmap(colors[~is_ventilator]), s=50, marker="x")
for sample in range(scores.shape[1]):
    h3, = ax.plot(
        scores[:, sample, 0],
        scores[:, sample, 1],
        color = "grey",
    )
ax.set_xlabel("PC - 0")
ax.set_ylabel("PC - 1")
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(min_age, max_age)),
    label = "Age",
    ax = ax,
)
fig.legend([h1, h2, h3], ["Ventilator", "Not Ventilator", "Trajectory"])

# %%
# Compute bootstrap
bs_results, bs = results.bootstrap(nbootstraps=200)
print(pandas.DataFrame(bs_results))

# %%
