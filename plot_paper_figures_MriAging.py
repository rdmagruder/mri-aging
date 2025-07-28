import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import re
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,               # Minimum 8pt for IEEE; 10pt preferred if space allows
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8
})

# repo path is the directory of this file
repo_dir = os.path.dirname(os.path.realpath(__file__))

# Load data
data = pd.read_excel(os.path.join(repo_dir, 'opencap_merged.xlsx'))

# get Radial_Diffusivity and Volume for each muscle, for each Subject and Timepoint
muscles = ['RF', 'VI', 'VM', 'VL']
# define timepoints from data
timepoints = np.unique(data['Timepoint'])
velocities = [0, -60, -45, 90, 120]
# define subjects from data - it is a string
subjects = pd.unique(data['Subject'])
# remove nan from timepoints and subjects
timepoints = timepoints[~np.isnan(timepoints)]
subjects = subjects[~pd.isnull(subjects)]
# get all columns that start with peak_kem_stand
peak_kem_stand_columns = [col for col in data.columns if col.startswith('peak_kem_stand')]
peak_kem_sit_columns = [col for col in data.columns if col.startswith('peak_kem_sit')]
peak_hfm_stand_columns = [col for col in data.columns if col.startswith('peak_hfm_stand')]
peak_hfm_sit_columns = [col for col in data.columns if col.startswith('peak_hfm_sit')]
peak_force_sit_columns = [col for col in data.columns if col.startswith('peak_force_sit')]
peak_force_stand_columns = [col for col in data.columns if col.startswith('peak_force_stand')]
peak_torso_ang_vel_columns = [col for col in data.columns if re.match(r'^torso_ang_vel[1-3]$', col)]
torso_orientation_liftoff_columns = [col for col in data.columns if re.match(r'^torso_ori_liftoff[1-3]$', col)]

# create a dictionary to store the data for each muscle
data_dict = {'Peak KEM Stand': np.zeros((len(subjects), len(timepoints))),
             'Peak KEM Sit': np.zeros((len(subjects), len(timepoints))),
             'Peak HFM Stand': np.zeros((len(subjects), len(timepoints))),
             'Peak HFM Sit': np.zeros((len(subjects), len(timepoints))),
             'Peak Force Sit': np.zeros((len(subjects), len(timepoints))),
             'Peak Force Stand': np.zeros((len(subjects), len(timepoints))),
             'Volume total': np.zeros((len(subjects), len(timepoints))),
             'contractile_volume': np.zeros((len(subjects), len(timepoints))),
             'Age': np.zeros((len(subjects), len(timepoints))),
             'Sex': np.full((len(subjects), len(timepoints)), None, dtype=object),
             'Weight': np.zeros((len(subjects), len(timepoints))),
             'Height': np.zeros((len(subjects), len(timepoints))),
             'STS_time': np.zeros((len(subjects), len(timepoints))),
             'Peak Torso Ang Velocity': np.zeros((len(subjects), len(timepoints))),
             'Torso Orientation at Liftoff': np.zeros((len(subjects), len(timepoints)))}

for muscle in muscles:
    data_dict[muscle] = {'Radial Diffusivity': np.zeros((len(subjects), len(timepoints))),
                         'Volume': np.zeros((len(subjects), len(timepoints))),
                         'Fat Fraction': np.zeros((len(subjects), len(timepoints))),
                         'CSA': np.zeros((len(subjects), len(timepoints))),
                         'Torque': np.zeros((len(subjects), len(timepoints), len(velocities)))}

    for i, subject in enumerate(subjects):
        for j, timepoint in enumerate(timepoints):
            data_dict[muscle]['Radial Diffusivity'][i, j] = \
            data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)][
                'Radial_Diffusivity'].mean()
            data_dict[muscle]['Volume'][i, j] = \
            data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)][
                'Volume'].mean()
            data_dict[muscle]['Fat Fraction'][i, j] = \
            data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)][
                'Fat_Fraction'].mean()
            data_dict[muscle]['CSA'][i, j] = \
            data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)][
                'CSA'].mean()
            for k, velocity in enumerate(velocities):
                data_dict[muscle]['Torque'][i, j, k] = data[
                    (data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle) & (
                                data['Velocity'] == velocity)]['Torque'].mean()

            # get the peak_kem_stand columns and average for this subject and timepoint - average over whole matrix
            data_dict['Peak KEM Stand'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_kem_stand_columns])
            data_dict['Peak KEM Sit'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_kem_sit_columns])
            data_dict['Peak HFM Stand'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_hfm_stand_columns])
            data_dict['Peak HFM Sit'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_hfm_sit_columns])
            data_dict['Peak Force Sit'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_force_sit_columns])
            data_dict['Peak Force Stand'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_force_stand_columns])
            data_dict['Peak Torso Ang Velocity'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 peak_torso_ang_vel_columns])
            data_dict['Torso Orientation at Liftoff'][i, j] = np.mean(
                [data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in
                 torso_orientation_liftoff_columns])

            # Sum the volumes for each muscle
            if muscle == muscles[-1]:
                data_dict['Volume total'][i, j] = np.sum(data_dict[muscle]['Volume'][i, j] for muscle in muscles)
                data_dict['contractile_volume'][i, j] = np.sum(
                    [data_dict[muscle]['Volume'][i, j] * (1 - data_dict[muscle]['Fat Fraction'][i, j] / 100) for muscle
                     in muscles])

                # get the age for this subject and timepoint
                data_dict['Age'][i, j] = \
                data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Age'].values[0]
                # get the weight for this subject and timepoint
                data_dict['Weight'][i, j] = \
                data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Weight'].values[0]
                # get the height for this subject and timepoint
                data_dict['Height'][i, j] = \
                data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Height'].values[0]
                # get the STS_time for this subject and timepoint
                data_dict['STS_time'][i, j] = \
                data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['STS_stopwatch'].values[0]
                sex_value = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Sex']
                data_dict['Sex'][i, j] = sex_value.values[0]


# for each subject and timepoint, create a weighted average across muscles weighted by volume
# create a dictionary to store the weighted average for each subject and timepoint
weighted_diffusivity = np.zeros((len(subjects), len(timepoints)))
weighted_fat_fraction = np.zeros((len(subjects), len(timepoints)))
for i, subject in enumerate(subjects):
    for j, timepoint in enumerate(timepoints):
        total_volume = np.sum([data_dict[muscle]['Volume'][i, j] for muscle in muscles])
        weighted_diffusivity[i, j] = np.sum(
            [data_dict[muscle]['Radial Diffusivity'][i, j] * data_dict[muscle]['Volume'][i, j] / total_volume for muscle
             in muscles])
        weighted_fat_fraction[i, j] = np.sum(
            [data_dict[muscle]['Fat Fraction'][i, j] * data_dict[muscle]['Volume'][i, j] / total_volume for muscle in
             muscles])

# Second timepoint
timepoint_to_plot = 1
velocity_to_plot = 0  # 0, 1, 2, 3, 4 corresponds to 0, -60, -45, 90, 120
remove_nans = True
color_by_sex = True

radial_diffusivity = weighted_diffusivity[:, timepoint_to_plot]
fat_fraction = weighted_fat_fraction[:, timepoint_to_plot]
volume_total = data_dict['Volume total'][:, timepoint_to_plot]
contractile_volume = data_dict['contractile_volume'][:, timepoint_to_plot]
peak_kem_stand = data_dict['Peak KEM Stand'][:, timepoint_to_plot]
peak_kem_sit = data_dict['Peak KEM Sit'][:, timepoint_to_plot]
peak_hfm_stand = data_dict['Peak HFM Stand'][:, timepoint_to_plot]
peak_hfm_sit = data_dict['Peak HFM Sit'][:, timepoint_to_plot]
peak_force_stand = data_dict['Peak Force Stand'][:, timepoint_to_plot]
peak_force_sit = data_dict['Peak Force Sit'][:, timepoint_to_plot]
peak_torso_ang_vel = data_dict['Peak Torso Ang Velocity'][:, timepoint_to_plot]
torso_orientation_liftoff = data_dict['Torso Orientation at Liftoff'][:, timepoint_to_plot]
age = data_dict['Age'][:, timepoint_to_plot]
weight = data_dict['Weight'][:, timepoint_to_plot]
height = data_dict['Height'][:, timepoint_to_plot]
STS_time = data_dict['STS_time'][:, timepoint_to_plot]
torque0 = data_dict[muscles[0]]['Torque'][:, timepoint_to_plot, 0]
torque1 = data_dict[muscles[0]]['Torque'][:, timepoint_to_plot, 1]
torque2 = data_dict[muscles[0]]['Torque'][:, timepoint_to_plot, 2]
torque3 = data_dict[muscles[0]]['Torque'][:, timepoint_to_plot, 3]
torque4 = data_dict[muscles[0]]['Torque'][:, timepoint_to_plot, 4]

# see if theres a nan in the row for either radial_diffusivity or peak_kem_stand. Remove that row from both vectors if theres a nan in that row of either vector
no_nans = ~(np.isnan(peak_kem_stand) | np.isnan(radial_diffusivity))
radial_diffusivity = radial_diffusivity[no_nans]
peak_kem_stand = -np.array(peak_kem_stand)[no_nans]
peak_kem_sit = -np.array(peak_kem_sit)[no_nans]
peak_hfm_stand = -np.array(peak_hfm_stand)[no_nans]
peak_hfm_sit = -np.array(peak_hfm_sit)[no_nans]
volume_total = volume_total[no_nans]
contractile_volume = contractile_volume[no_nans]
fat_fraction = fat_fraction[no_nans]
age = age[no_nans]
weight = weight[no_nans]
height = height[no_nans]
STS_time = STS_time[no_nans]
peak_force_stand = peak_force_stand[no_nans]
peak_force_sit = peak_force_sit[no_nans]
peak_torso_ang_vel = peak_torso_ang_vel[no_nans]
torso_orientation_liftoff = torso_orientation_liftoff[no_nans]
torque0 = torque0[no_nans]
torque1 = torque1[no_nans]
torque2 = torque2[no_nans]
torque3 = torque3[no_nans]
torque4 = torque4[no_nans]
torques = np.column_stack((torque0, torque1, torque2, torque3, torque4))
sex_vector = np.array(data_dict['Sex'][:, timepoint_to_plot])[no_nans]


if color_by_sex:
    colors = ['#28aeed' if sex == 'M' else '#FFD100' for sex in sex_vector]
else:
    colors = '#28aeed'


# multiple regression model to predict peak_kem_stand from some number of inputs

# create a design matrix
radial_diffusivity_standardized = (radial_diffusivity - np.mean(radial_diffusivity)) / np.std(radial_diffusivity)
volume_total_standardized = (volume_total - np.mean(volume_total)) / np.std(volume_total)
contractile_volume_standardized = (contractile_volume - np.mean(contractile_volume)) / np.std(contractile_volume)

# make weight * height in kg*cm but convert to N*m
weight_height = (weight * height) / 100 * 9.81  # convert weight to N and height to cm

# # X = np.column_stack((torque0, torso_orientation_liftoff, peak_torso_ang_vel)) # only torque is sig
# X = np.column_stack((torque0, torque1, torque2, torque3, torque4))
# X = sm.add_constant(X)
# # fit the model
# model = sm.OLS(peak_kem_stand, X)
# results = model.fit()
# # print the results
# print(results.summary())

# compute correlation between peak_kem_stand and radial_diffusivity
# x = radial_diffusivity_standardized + volume_total_standardized
# x = radial_diffusivity_standardized + contractile_volume_standardized
x = radial_diffusivity_standardized + volume_total_standardized

# Build the DataFrame for regression
df = pd.DataFrame({
    'KEM': peak_kem_stand,
    'Volume': volume_total_standardized,
    'RD': radial_diffusivity_standardized,
    'Composite': x,
    'Age': age,
    'Sex': [1 if s == 'M' else 0 for s in sex_vector],  # encode sex: M=1, F=0
})




# Redo it but with a black line for the regression line with automatic confidence intervals using seaborn
# remove grid
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3))

# Plot 1: volume_total vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=volume_total_standardized, ax=ax[0], color='#28aeed', scatter_kws={'s': 20},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[0].set_xlabel('Knee Extension Moment (Nm)')
ax[0].set_ylabel('Volume Total (Standardized)')
correlation1, p_value1 = pearsonr(peak_kem_stand, volume_total_standardized)
p_str = f"p={p_value1:.3f}".replace("p=0", "p=") if p_value1 >= 0.001 else "p<.001"
ax[0].text(0.5, 1, f'r={correlation1:.2f}, {p_str}', transform=ax[0].transAxes, ha='center')

# Plot 2: radial diffusivity vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=radial_diffusivity_standardized, ax=ax[1], color='#28aeed', scatter_kws={'s': 20},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[1].set_xlabel('Knee Extension Moment (Nm)')
ax[1].set_ylabel('Radial Diffusivity (Standardized)')
correlation2, p_value2 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
# ax[1].text(0.5, 1, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
p_str = f"p={p_value2:.3f}".replace("p=0", "p=") if p_value2 >= 0.001 else "p<.001"
ax[1].text(0.5, 1, f'r={correlation2:.2f}, {p_str}', transform=ax[1].transAxes, ha='center')
# Plot 3: x vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=x, ax=ax[2], color='#28aeed', scatter_kws={'s': 20},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[2].set_xlabel('Knee Extension Moment (Nm)')
ax[2].set_ylabel('Radial Diffusivity + Volume Total (Standardized)')
correlation3, p_value3 = pearsonr(peak_kem_stand, x)
# ax[2].text(0.5, 1, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
p_str = f"p={p_value3:.3f}".replace("p=0", "p=") if p_value3 >= 0.001 else "p<.001"
ax[2].text(0.5, 1, f'r={correlation3:.2f}, {p_str}', transform=ax[2].transAxes, ha='center')
# Despine all subplots (top and right only)
for axis in ax:
    sns.despine(ax=axis, top=True, right=True)
fig.subplots_adjust(wspace=0.3)  # Adjust space between subplots
# Save the figure
plt.savefig(os.path.join(repo_dir, f'finalFigures/KEM_vs_MRI_timepoint{timepoint_to_plot}_regression.svg'),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()



# Metrics vs x
correlation_KEM, p_value_KEM = pearsonr(peak_kem_stand, x)
correlation_STS, p_value_STS = pearsonr(STS_time, x)
correlation_orientation, p_value_orientation = pearsonr(torso_orientation_liftoff, x)
correlation_ang_vel, p_value_ang_vel = pearsonr(peak_torso_ang_vel, x)
print('Peak KEM Stand and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_KEM, p_value_KEM))
print('STS Time and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_STS, p_value_STS))
print('Torso Orientation at Liftoff and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_orientation, p_value_orientation))
print('Peak Torso Angular Velocity and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_ang_vel, p_value_ang_vel))
print()

# Metrics vs standardized volume_total
correlation_KEM, p_value_KEM = pearsonr(peak_kem_stand, volume_total_standardized)
correlation_STS, p_value_STS = pearsonr(STS_time, volume_total_standardized)
correlation_orientation, p_value_orientation = pearsonr(torso_orientation_liftoff, volume_total_standardized)
correlation_ang_vel, p_value_ang_vel = pearsonr(peak_torso_ang_vel, volume_total_standardized)
print('Peak KEM Stand and Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_KEM, p_value_KEM))
print('STS Time and Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_STS, p_value_STS))
print('Torso Orientation at Liftoff and Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_orientation, p_value_orientation))
print('Peak Torso Angular Velocity and Volume Total: r= {:.2f}, p= {:.2e}'.format(
    correlation_ang_vel, p_value_ang_vel))
print()

# Metrics vs standardized radial_diffusivity
correlation_KEM, p_value_KEM = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
correlation_STS, p_value_STS = pearsonr(STS_time, radial_diffusivity_standardized)
correlation_orientation, p_value_orientation = pearsonr(torso_orientation_liftoff, radial_diffusivity_standardized)
correlation_ang_vel, p_value_ang_vel = pearsonr(peak_torso_ang_vel, radial_diffusivity_standardized)
print('Peak KEM Stand and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(correlation_KEM, p_value_KEM))
print('STS Time and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(correlation_STS, p_value_STS))
print('Torso Orientation at Liftoff and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(
    correlation_orientation, p_value_orientation))
print('Peak Torso Angular Velocity and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(
    correlation_ang_vel, p_value_ang_vel))
print()


# correlations of kem vs volume, rd, and x
correlation_KEM_volume, p_value_KEM_volume = pearsonr(peak_kem_stand, volume_total_standardized)
correlation_KEM_rd, p_value_KEM_rd = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
correlation_KEM_x, p_value_KEM_x = pearsonr(peak_kem_stand, x)

# correlations of isometric torque vs volume, rd, and x
correlation_torque0_volume, p_value_torque0_volume = pearsonr(torque0, volume_total_standardized)
correlation_torque0_rd, p_value_torque0_rd = pearsonr(torque0, radial_diffusivity_standardized)
correlation_torque0_x, p_value_torque0_x = pearsonr(torque0, x)

# correlations of isokinetics (torque4) vs volume, rd, and x
correlation_torque4_volume, p_value_torque4_volume = pearsonr(torque4, volume_total_standardized)
correlation_torque4_rd, p_value_torque4_rd = pearsonr(torque4, radial_diffusivity_standardized)
correlation_torque4_x, p_value_torque4_x = pearsonr(torque4, x)

# print the correlations
print('Peak KEM Stand and Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_KEM_volume, p_value_KEM_volume))
print('Peak KEM Stand and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(correlation_KEM_rd, p_value_KEM_rd))
print('Peak KEM Stand and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_KEM_x, p_value_KEM_x))
print('Isometric Torque and Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_torque0_volume, p_value_torque0_volume))
print('Isometric Torque and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(correlation_torque0_rd, p_value_torque0_rd))
print('Isometric Torque and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_torque0_x, p_value_torque0_x))
print('Isokinetic Torque and Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_torque4_volume, p_value_torque4_volume))
print('Isokinetic Torque and Radial Diffusivity: r= {:.2f}, p= {:.2e}'.format(correlation_torque4_rd, p_value_torque4_rd))
print('Isokinetic Torque and Radial Diffusivity + Volume Total: r= {:.2f}, p= {:.2e}'.format(correlation_torque4_x, p_value_torque4_x))

correlation_STS_volume, p_value_STS_volume = pearsonr(STS_time, volume_total_standardized)
correlation_STS_rd, p_value_STS_rd = pearsonr(STS_time, radial_diffusivity_standardized)
correlation_STS_x, p_value_STS_x = pearsonr(STS_time, x)
correlation_orientation_volume, p_value_orientation_volume = pearsonr(torso_orientation_liftoff, volume_total_standardized)
correlation_orientation_rd, p_value_orientation_rd = pearsonr(torso_orientation_liftoff, radial_diffusivity_standardized)
correlation_orientation_x, p_value_orientation_x = pearsonr(torso_orientation_liftoff, x)
correlation_ang_vel_volume, p_value_ang_vel_volume = pearsonr(peak_torso_ang_vel, volume_total_standardized)
correlation_ang_vel_rd, p_value_ang_vel_rd = pearsonr(peak_torso_ang_vel, radial_diffusivity_standardized)
correlation_ang_vel_x, p_value_ang_vel_x = pearsonr(peak_torso_ang_vel, x)

# make a bar plot of the correlations. the groups are Peak KEM Stand, Isometric Torque, and Isokinetic Torque (120 deg/s). each group has three bars: Volume Total, Radial Diffusivity, and Radial Diffusivity + Volume Total
# Define the data
groups = ['Knee Extension\nMoment (Nm)', 'Isometric\nTorque (Nm)', '120 deg/s Isokinetic\nTorque (Nm)']
correlations = [
    [correlation_KEM_volume, correlation_KEM_rd, correlation_KEM_x],
    [correlation_torque0_volume, correlation_torque0_rd, correlation_torque0_x],
    [correlation_torque4_volume, correlation_torque4_rd, correlation_torque4_x]
]
# Create the bar plot
loc = np.arange(len(groups))*.9  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(4, 3))
bars1 = ax.bar(loc - width, [corr[0] for corr in correlations], width, label='Volume Total', color='#FFD100')
bars2 = ax.bar(loc, [corr[1] for corr in correlations], width, label='Radial Diffusivity', color='#43d1ad')
bars3 = ax.bar(loc + width, [corr[2] for corr in correlations], width, label='Radial Diffusivity + Volume Total', color='#b4656f')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Correlation Coefficient')
ax.set_xticks(loc)
ax.set_xticklabels(groups)
ax.set_ylim(0, 1)  # y-axis starts at bottom of lowest bar
ax.axhline(0, color='black', linewidth=1)  # ensure x-axis is shown at y=0
ax.tick_params(axis='x', length=0)  # hide x tick marks but keep labels
# Add a * to the top of the bars if the p-value is less than 0.05
def add_significance_bars(bars, p_values):
    for bar, p_value in zip(bars, p_values):
        if p_value < 0.05:
            height = bar.get_height()
            ax.annotate('*',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=16, color='black',)

add_significance_bars(bars1, [p_value_KEM_volume, p_value_torque0_volume, p_value_torque4_volume])
add_significance_bars(bars2, [p_value_KEM_rd, p_value_torque0_rd, p_value_torque4_rd])
add_significance_bars(bars3, [p_value_KEM_x, p_value_torque0_x, p_value_torque4_x])

# move the legend outside the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
# Adjust layout
plt.tight_layout()
sns.despine(ax=ax, top=True, right=True)# Save the figure
plt.savefig(os.path.join(repo_dir, 'finalFigures/correlation_barplot_torque_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

# make a bar plot with groups KEM, isomet, isokin, STS, orientation, ang_vel with three bars for each group: volume_total, radial_diffusivity, and x (radial diffusivity + volume total)
# Define the data for the bar plot
groups = ['Knee Extension\nMoment (Nm)\nOpenCap', 'STS\nTime (s)', 'Torso\nAngle (deg)',
            'Torso Angular\nVelocity (deg/s)']
correlations = [
    [correlation_KEM_volume, correlation_KEM_rd, correlation_KEM_x],
    [correlation_STS_volume, correlation_STS_rd, correlation_STS_x],
    [correlation_orientation_volume, correlation_orientation_rd, correlation_orientation_x],
    [correlation_ang_vel_volume, correlation_ang_vel_rd, correlation_ang_vel_x]
]
# Create the bar plot
loc = np.arange(len(groups))*.9  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(4, 3))
bars1 = ax.bar(loc - width, [corr[0] for corr in correlations], width, label='Volume Total', color='#FFD100')
bars2 = ax.bar(loc, [corr[1] for corr in correlations], width, label='Radial Diffusivity', color='#43d1ad')
bars3 = ax.bar(loc + width, [corr[2] for corr in correlations], width, label='Radial Diffusivity + Volume Total', color='#b4656f')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Correlation Coefficient')
ax.set_xticks(loc)
ax.set_xticklabels(groups)
ax.set_ylim(None, 1)  # y-axis starts at bottom of lowest bar
ax.axhline(0, color='black', linewidth=1)  # ensure x-axis is shown at y=0
ax.tick_params(axis='x', length=0)  # hide x tick marks but keep labels
# Add a * to the top of the bars if the p-value is less than 0.05
add_significance_bars(bars1, [p_value_KEM_volume,
                                p_value_STS_volume, p_value_orientation_volume, p_value_ang_vel_volume])
add_significance_bars(bars2, [p_value_KEM_rd,
                                p_value_STS_rd, p_value_orientation_rd, p_value_ang_vel_rd])
add_significance_bars(bars3, [p_value_KEM_x,
                                p_value_STS_x, p_value_orientation_x, p_value_ang_vel_x])
# move the legend outside the plot
ax.set_ylim(None, 1)  # Set y-axis limits for better visibility
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
# Adjust layout
plt.tight_layout()
sns.despine(ax=ax, top=True, right=True, bottom=True)
# Save the figure
plt.savefig(os.path.join(repo_dir, 'finalFigures/correlation_barplot_sts_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

# Group names and colors
mri_groups = ['Volume Total', 'Radial Diffusivity', 'Volume + Radial Diffusivity']
metrics = ['Knee Extension\nMoment (Nm)\nOpenCap', '120 deg/s Isokinetic\nTorque (Nm)\nDynamometry', 'Isometric\nTorque (Nm)\nDynamometry']
colors = ['#FFD100', '#43d1ad', '#b4656f']

# Correlations grouped by MRI metric
correlations_by_mri = [
    [correlation_KEM_volume, correlation_torque4_volume, correlation_torque0_volume],
    [correlation_KEM_rd, correlation_torque4_rd, correlation_torque0_rd],
    [correlation_KEM_x, correlation_torque4_x, correlation_torque0_x],
]

# Corresponding p-values
p_values_by_mri = [
    [p_value_KEM_volume, p_value_torque4_volume, p_value_torque0_volume],
    [p_value_KEM_rd, p_value_torque4_rd, p_value_torque0_rd],
    [p_value_KEM_x, p_value_torque4_x, p_value_torque0_x],
]

# Plot setup
width = 0.2
loc = np.arange(len(metrics))  # KEM, isokin, isomet

fig, ax = plt.subplots(figsize=(4, 3))

bars = []
for i in range(len(mri_groups)):
    bars.append(
        ax.bar(loc + (i - 1) * width, correlations_by_mri[i], width,
               label=mri_groups[i], color=colors[i])
    )

# Add significance stars
for bar_group, p_values in zip(bars, p_values_by_mri):
    add_significance_bars(bar_group, p_values)

# Labels and style
ax.set_ylabel('Correlation Coefficient with MRI Metrics')
ax.set_xticks(loc)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.axhline(0, color='black', linewidth=1)
ax.tick_params(axis='x', length=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)

sns.despine(ax=ax, top=True, right=True, bottom=True)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(repo_dir, f'finalFigures/correlation_barplot_torque_grouped_by_mri_timepoint{timepoint_to_plot}.svg'),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()


# Setup
mri_metrics = ['Volume Total', 'Radial Diffusivity', 'Volume + RD']
metrics = ['KEM', 'Torso Ang Vel', 'Torso Ang', 'STS Time']
colors = ['#c85250', '#5292c6', '#e49b3a', '#59b87f']  # One color per metric

# Correlation and p-values grouped by MRI metric
correlations_by_mri = [
    [correlation_KEM_volume, correlation_ang_vel_volume, correlation_orientation_volume, correlation_STS_volume],
    [correlation_KEM_rd, correlation_ang_vel_rd, correlation_orientation_rd, correlation_STS_rd],
    [correlation_KEM_x, correlation_ang_vel_x, correlation_orientation_x, correlation_STS_x],
]

p_values_by_mri = [
    [p_value_KEM_volume, p_value_ang_vel_volume, p_value_orientation_volume, p_value_STS_volume],
    [p_value_KEM_rd, p_value_ang_vel_rd, p_value_orientation_rd, p_value_STS_rd],
    [p_value_KEM_x, p_value_ang_vel_x, p_value_orientation_x, p_value_STS_x],
]

# Plot setup
width = 0.2
loc = np.arange(len(mri_metrics))  # Volume, RD, Volume+RD

fig, ax = plt.subplots(figsize=(5.2, 3))
bars = []

# Plot 4 bars per group (one for each metric) across MRI types
for i in range(4):  # 4 metrics
    bar = ax.bar(loc + (i - 1.5) * width, [corr[i] for corr in correlations_by_mri],
                 width, label=metrics[i], color=colors[i])
    bars.append(bar)

# Add significance stars per bar
for i in range(4):
    add_significance_bars(bars[i], [p[i] for p in p_values_by_mri])

# Final touches
ax.set_ylabel('Correlation Coefficient')
ax.set_xticks(loc)
ax.set_xticklabels(mri_metrics)
ax.set_ylim(None, 1)  # y-axis starts at bottom of lowest bar
ax.axhline(0, color='black', linewidth=1)  # ensure x-axis is shown at y=0
ax.tick_params(axis='x', length=0)  # hide x tick marks but keep labelsax.axhline(0, color='black', linewidth=1)
ax.tick_params(axis='x', length=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=2)

sns.despine(ax=ax, top=True, right=True, bottom=True)
plt.tight_layout()

# Save
plt.savefig(os.path.join(repo_dir, f'finalFigures/correlation_barplot_grouped_by_mri_timepoint{timepoint_to_plot}.svg'),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()



colors = '#28aeed'
isokin_by_isomet = torque4/torque0
KEM_by_isomet = peak_kem_stand/torque0
# make a 2x3 subplot. y axis will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). row 1 will be isokin_by_isomet, row 2 will be KEM_by_isomet

fig, ax = plt.subplots(2, 2, figsize=(3.6, 4))
plots = [
    (ax[0, 0], isokin_by_isomet, volume_total_standardized, 'Volume Total (Standardized)', 'Isokinetic / Isometric Torque'),
    (ax[0, 1], isokin_by_isomet, radial_diffusivity_standardized, 'Radial Diffusivity (Standardized)', 'Isokinetic / Isometric Torque'),
    (ax[1, 0], KEM_by_isomet, volume_total_standardized, 'Volume Total (Standardized)', 'KEM / Isometric Torque'),
    (ax[1, 1], KEM_by_isomet, radial_diffusivity_standardized, 'Radial Diffusivity (Standardized)', 'KEM / Isometric Torque')
]

for i, (axis, x_value, y, ylabel, xlabel) in enumerate(plots):
    correlation, p_value = pearsonr(x_value, y)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    p_str = f"p={p_value:.3f}".replace("p=0", "p=") if p_value >= 0.001 else "p<.001"
    axis.text(0.5, 1, f'r={correlation:.2f}, {p_str}', transform=axis.transAxes, ha='center')

    if p_value < 0.05:
        sns.regplot(x=x_value, y=y, ax=axis, scatter=True, color=colors, ci=95, line_kws={'color': 'black'}, scatter_kws={'s': 20})
    else:
        axis.scatter(x_value, y, c=colors, s=20)
plt.tight_layout()
for axis in ax.flat:
    sns.despine(ax=axis, trim=False)
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'finalFigures/ratios_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

colors = ['#28aeed' if sex == 'M' else '#FFD100' for sex in sex_vector]


df = pd.DataFrame({
    'age': age,
    'sex': [1 if s == 'F' else 0 for s in sex_vector],  # F=1, M=0
    'volume': volume_total_standardized,
    'radial_diffusivity': radial_diffusivity_standardized,
    'composite_x': x,
    'isometric_BwH': torque0 / weight_height * 100,  # Convert to percentage of weight-height
    'isokinetic_BwH': torque4 / weight_height * 100,  # Convert to percentage of weight-height
    'kem_BwH': peak_kem_stand / weight_height * 100, # Convert to percentage of weight-height
    'isometric': torque0,
    'isokinetic': torque4,
    'kem': peak_kem_stand,
})

# scaling outcomes
df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
df['isometric_std'] = (df['isometric'] - df['isometric'].mean()) / df['isometric'].std(ddof=0)
df['isokinetic_std'] = (df['isokinetic'] - df['isokinetic'].mean()) / df['isokinetic'].std(ddof=0)
df['kem_std'] = (df['kem'] - df['kem'].mean()) / df['kem'].std(ddof=0)
df['isometric_std_BwH'] = (df['isometric_BwH'] - df['isometric_BwH'].mean()) / df['isometric_BwH'].std(ddof=0)
df['isokinetic_std_BwH'] = (df['isokinetic_BwH'] - df['isokinetic_BwH'].mean()) / df['isokinetic_BwH'].std(ddof=0)
df['kem_std_BwH'] = (df['kem_BwH'] - df['kem_BwH'].mean()) / df['kem_BwH'].std(ddof=0)

# Labels and targets
outcomes = ['volume', 'radial_diffusivity', 'composite_x', 'isometric_std', 'isokinetic_std', 'kem_std',
            'isometric_std_BwH', 'isokinetic_std_BwH', 'kem_std_BwH']
outcome_labels = ['Volume', 'Radial Diffusivity', 'Composite',
                  'Isometric Torque (Standardized)', '120 deg/s Isokinetic Torque (Standardized)', 'Peak KEM Stand (Standardized)',
                  'Isometric Torque (%BwH)', '120 deg/s Isokinetic Torque (%BwH)', 'Peak KEM Stand (%BwH)']

# Table to store results
results = []

# Function to compute partial eta squared
def compute_partial_eta_sq(full_model, reduced_model):
    ssr_full = full_model.ssr
    ssr_reduced = reduced_model.ssr
    ss_effect = ssr_reduced - ssr_full
    return ss_effect / (ss_effect + ssr_full)

# Run models and collect values
for outcome, label in zip(outcomes, outcome_labels):
    y = df[outcome]
    X = sm.add_constant(df[['age_std', 'sex']])
    model = sm.OLS(y, X).fit()

    model_age = sm.OLS(y, sm.add_constant(df[['sex']])).fit()
    model_sex = sm.OLS(y, sm.add_constant(df[['age_std']])).fit()

    eta_sq_age = compute_partial_eta_sq(model, model_age)
    eta_sq_sex = compute_partial_eta_sq(model, model_sex)

    results.append({
        'Outcome': label,
        'Age β (95% CI)': f"{model.params['age_std']:.2f} ({model.conf_int().loc['age_std', 0]:.2f}, {model.conf_int().loc['age_std', 1]:.2f})",
        'Age p': f"{model.pvalues['age_std']:.3f}",
        'Age η²p': f"{eta_sq_age:.3f}",
        'Sex β (95% CI)': f"{model.params['sex']:.2f} ({model.conf_int().loc['sex', 0]:.2f}, {model.conf_int().loc['sex', 1]:.2f})",
        'Sex p': f"{model.pvalues['sex']:.3f}",
        'Sex η²p': f"{eta_sq_sex:.3f}",
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_excel(os.path.join(repo_dir, 'finalFigures/regression_results_timepoint{}_BwH_Norm.xlsx'.format(timepoint_to_plot)),
                     index=False)


from sklearn.linear_model import LinearRegression
# targets = ['volume', 'radial_diffusivity', 'composite_x']
# target_labels = ['Volume', 'Radial Diffusivity', 'Composite']
#
# predictors = ['isometric', 'isokinetic', 'kem']
# predictor_labels = ['Isometric Torque (%BwH)', '120 deg/s Isokinetic Torque (%BwH)', 'Peak KEM Stand (%BwH)']
#
# results = []
#
# for predictor, pred_label in zip(predictors, predictor_labels):
#     for target, target_label in zip(targets, target_labels):
#         # Raw correlation
#         raw_r, raw_p = pearsonr(df[predictor], df[target])
#
#         # Adjusted (partial) correlation: remove age and sex effects
#         covariates = df[['age', 'sex']]
#
#         # Regress out covariates from both predictor and target
#         pred_resid = df[predictor] - LinearRegression().fit(covariates, df[predictor]).predict(covariates)
#         target_resid = df[target] - LinearRegression().fit(covariates, df[target]).predict(covariates)
#
#         # Correlate residuals (partial correlation)
#         adj_r, adj_p = pearsonr(pred_resid, target_resid)
#
#         results.append({
#             'Predictor': pred_label,
#             'Outcome': target_label,
#             'Raw r': f"{raw_r:.3f}",
#             'Raw p': f"{raw_p:.3f}",
#             'Adjusted r (age+sex)': f"{adj_r:.3f}",
#             'Adjusted p (age+sex)': f"{adj_p:.3f}"
#         })
#
# results_df = pd.DataFrame(results)
# print(results_df.to_string(index=False))
#
# # Save
# results_df.to_excel(os.path.join(repo_dir, f'finalFigures/correlation_results_partial_age_sex_timepoint{timepoint_to_plot}.xlsx'), index=False)
#
#
# predictors = {
#     'Knee Extension\nMoment (Nm)': peak_kem_stand,
#     'Isometric\nTorque (Nm)': torque0,
#     '120 deg/s Isokinetic\nTorque (Nm)': torque4
# }
# outcomes = {
#     'Volume Total': volume_total_standardized,
#     'Radial Diffusivity': radial_diffusivity_standardized,
#     'Radial Diffusivity + Volume Total': x
# }
# covariates = np.column_stack((age, [1 if s == 'F' else 0 for s in sex_vector]))
#
# # Compute adjusted r (partial correlations)
# adj_r_matrix = []
# p_matrix = []
#
# for pred_vals in predictors.values():
#     row_adj_r = []
#     row_p = []
#     for out_vals in outcomes.values():
#         pred_resid = pred_vals - LinearRegression().fit(covariates, pred_vals).predict(covariates)
#         out_resid = out_vals - LinearRegression().fit(covariates, out_vals).predict(covariates)
#         r_adj, p_adj = pearsonr(pred_resid, out_resid)
#         row_adj_r.append(r_adj)
#         row_p.append(p_adj)
#     adj_r_matrix.append(row_adj_r)
#     p_matrix.append(row_p)
#
# # Plot
# loc = np.arange(len(predictors)) * 0.9
# width = 0.25
# fig, ax = plt.subplots(figsize=(4, 3))
# bars1 = ax.bar(loc - width, [row[0] for row in adj_r_matrix], width, label='Volume Total', color='#FFD100')
# bars2 = ax.bar(loc, [row[1] for row in adj_r_matrix], width, label='Radial Diffusivity', color='#43d1ad')
# bars3 = ax.bar(loc + width, [row[2] for row in adj_r_matrix], width, label='RD + Volume', color='#b4656f')
#
# ax.set_ylabel('Adjusted Correlation Coefficient')
# ax.set_xticks(loc)
# ax.set_xticklabels(predictors.keys())
# ax.set_ylim(-.25, 1)
# ax.axhline(0, color='black', linewidth=1)
# ax.tick_params(axis='x', length=0)
#
# def add_stars(bars, p_vals):
#     for bar, p in zip(bars, p_vals):
#         if p < 0.05:
#             height = bar.get_height()
#             ax.annotate('*', (bar.get_x() + bar.get_width()/2, height), ha='center', va='bottom', fontsize=16)
#
# add_stars(bars1, [row[0] for row in p_matrix])
# add_stars(bars2, [row[1] for row in p_matrix])
# add_stars(bars3, [row[2] for row in p_matrix])
#
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False)
# plt.tight_layout()
# sns.despine(ax=ax, top=True, right=True, bottom=True)
# plt.show()