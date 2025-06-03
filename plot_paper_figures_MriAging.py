import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import re
import seaborn as sns

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
    colors = ['blue' if sex == 'M' else 'orange' for sex in sex_vector]
else:
    colors = 'blue'


# multiple regression model to predict peak_kem_stand from some number of inputs
import statsmodels.api as sm

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

# # Figure 1
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# # Plot 1: volume_total vs peak_kem_stand
# ax[0].scatter(peak_kem_stand, volume_total_standardized, c=colors)
# ax[0].set_xlabel('Peak KEM Stand')
# ax[0].set_ylabel('Volume Total (Standardized)')
# correlation1, p_value1 = pearsonr(peak_kem_stand, volume_total_standardized)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#               ha='center')
# # Plot 8: radial diffusivity vs peak_kem_stand
# ax[1].scatter(peak_kem_stand, radial_diffusivity_standardized, c=colors)
# ax[1].set_xlabel('Peak KEM Stand')
# ax[1].set_ylabel('Radial Diffusivity (Standardized)')
# correlation2, p_value2 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#               ha='center')
# # Plot 9: x vs peak_kem_stand
# ax[2].scatter(peak_kem_stand, x, c=colors)
# ax[2].set_xlabel('Peak KEM Stand')
# ax[2].set_ylabel('Radial Diffusivity + Volume Total (Standardized)')
# correlation3, p_value3 = pearsonr(peak_kem_stand, x)
# ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'finalFigures/KEM_vs_MRI_timepoint{}_bySex.svg'.format(timepoint_to_plot)),
#             format='svg', dpi=300)
# plt.show()

# Redo it but with a black line for the regression line with automatic confidence intervals using seaborn
# remove grid
plt.style.use('default')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: volume_total vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=volume_total_standardized, ax=ax[0], color='blue', scatter_kws={'s': 50},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[0].set_xlabel('Peak KEM Stand')
ax[0].set_ylabel('Volume Total (Standardized)')
correlation1, p_value1 = pearsonr(peak_kem_stand, volume_total_standardized)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')

# Plot 2: radial diffusivity vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=radial_diffusivity_standardized, ax=ax[1], color='blue', scatter_kws={'s': 50},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[1].set_xlabel('Peak KEM Stand')
ax[1].set_ylabel('Radial Diffusivity (Standardized)')
correlation2, p_value2 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')

# Plot 3: x vs peak_kem_stand
sns.regplot(x=peak_kem_stand, y=x, ax=ax[2], color='blue', scatter_kws={'s': 50},
            line_kws={'color': 'black', 'linewidth': 2}, ci=95)
ax[2].set_xlabel('Peak KEM Stand')
ax[2].set_ylabel('Radial Diffusivity + Volume Total (Standardized)')
correlation3, p_value3 = pearsonr(peak_kem_stand, x)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')

# Despine all subplots (top and right only)
for axis in ax:
    sns.despine(ax=axis, top=True, right=True)

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
import matplotlib.pyplot as plt
# Define the data
groups = ['Peak KEM Stand', 'Isometric Torque', 'Isokinetic Torque (120 deg/s)']
correlations = [
    [correlation_KEM_volume, correlation_KEM_rd, correlation_KEM_x],
    [correlation_torque0_volume, correlation_torque0_rd, correlation_torque0_x],
    [correlation_torque4_volume, correlation_torque4_rd, correlation_torque4_x]
]
# Create the bar plot
loc = np.arange(len(groups))*.9  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(loc - width, [corr[0] for corr in correlations], width, label='Volume Total', color='blue')
bars2 = ax.bar(loc, [corr[1] for corr in correlations], width, label='Radial Diffusivity', color='orange')
bars3 = ax.bar(loc + width, [corr[2] for corr in correlations], width, label='Radial Diffusivity + Volume Total', color='green')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Correlations of Metrics with Radial Diffusivity and Volume Total')
ax.set_xticks(loc)
ax.set_xticklabels(groups)
ax.set_ylim(None, 1)  # Set y-axis limits for better visibility
# Add a * to the top of the bars if the p-value is less than 0.05
def add_significance_bars(bars, p_values):
    for bar, p_value in zip(bars, p_values):
        if p_value < 0.05:
            height = bar.get_height()
            ax.annotate('*',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

add_significance_bars(bars1, [p_value_KEM_volume, p_value_torque0_volume, p_value_torque4_volume])
add_significance_bars(bars2, [p_value_KEM_rd, p_value_torque0_rd, p_value_torque4_rd])
add_significance_bars(bars3, [p_value_KEM_x, p_value_torque0_x, p_value_torque4_x])

# move the legend outside the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False)
# Adjust layout
plt.tight_layout()
sns.despine(ax=ax, top=True, right=True)# Save the figure
plt.savefig(os.path.join(repo_dir, 'finalFigures/correlation_barplot_torque_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

# make a bar plot with groups KEM, isomet, isokin, STS, orientation, ang_vel with three bars for each group: volume_total, radial_diffusivity, and x (radial diffusivity + volume total)
# Define the data for the bar plot
groups = ['KEM', 'STS Time', 'Torso Orientation at Liftoff', 'Peak Torso Angular Velocity']
correlations = [
    [correlation_KEM_volume, correlation_KEM_rd, correlation_KEM_x],
    [correlation_STS_volume, correlation_STS_rd, correlation_STS_x],
    [correlation_orientation_volume, correlation_orientation_rd, correlation_orientation_x],
    [correlation_ang_vel_volume, correlation_ang_vel_rd, correlation_ang_vel_x]
]
# Create the bar plot
loc = np.arange(len(groups))*.9  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(loc - width, [corr[0] for corr in correlations], width, label='Volume Total', color='blue')
bars2 = ax.bar(loc, [corr[1] for corr in correlations], width, label='Radial Diffusivity', color='orange')
bars3 = ax.bar(loc + width, [corr[2] for corr in correlations], width, label='Radial Diffusivity + Volume Total', color='green')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Correlation Coefficient')
ax.set_title('Correlations of Metrics with Radial Diffusivity and Volume Total')
ax.set_xticks(loc)
ax.set_xticklabels(groups)
# Add a * to the top of the bars if the p-value is less than 0.05
add_significance_bars(bars1, [p_value_KEM_volume,
                                p_value_STS_volume, p_value_orientation_volume, p_value_ang_vel_volume])
add_significance_bars(bars2, [p_value_KEM_rd,
                                p_value_STS_rd, p_value_orientation_rd, p_value_ang_vel_rd])
add_significance_bars(bars3, [p_value_KEM_x,
                                p_value_STS_x, p_value_orientation_x, p_value_ang_vel_x])
# move the legend outside the plot
ax.set_ylim(None, 1)  # Set y-axis limits for better visibility
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), frameon=False)
# Adjust layout
plt.tight_layout()
sns.despine(ax=ax, top=True, right=True)
# Save the figure
plt.savefig(os.path.join(repo_dir, 'finalFigures/correlation_barplot_sts_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

colors = 'blue'
isokin_by_isomet = torque4/torque0
KEM_by_isomet = peak_kem_stand/torque0
# make a 2x3 subplot. y axis will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). row 1 will be isokin_by_isomet, row 2 will be KEM_by_isomet

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
plots = [
    (ax[0, 0], isokin_by_isomet, volume_total_standardized, 'Volume Total (Standardized)', 'Isokinetic / Isometric Torque'),
    (ax[0, 1], isokin_by_isomet, radial_diffusivity_standardized, 'Radial Diffusivity (Standardized)', 'Isokinetic / Isometric Torque'),
    (ax[1, 0], KEM_by_isomet, volume_total_standardized, 'Volume Total (Standardized)', 'KEM / Isometric Torque'),
    (ax[1, 1], KEM_by_isomet, radial_diffusivity_standardized, 'Radial Diffusivity (Standardized)', 'KEM / Isometric Torque')
]

for i, (axis, x, y, ylabel, xlabel) in enumerate(plots):
    correlation, p_value = pearsonr(x, y)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.text(0.5, 0.9, f'r= {correlation:.2f}, p= {p_value:.2e}', transform=axis.transAxes, ha='center')

    if p_value < 0.05:
        sns.regplot(x=x, y=y, ax=axis, scatter=True, color=colors, ci=95, line_kws={'color': 'black'})
    else:
        axis.scatter(x, y, c=colors)
plt.tight_layout()
for axis in ax.flat:
    sns.despine(ax=axis, trim=False)
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'finalFigures/ratios_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()

colors = ['blue' if sex == 'M' else 'orange' for sex in sex_vector]

# # make a 2x3 subplot. y axis row 1 will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). row 2 will be torque0, torque4, and KEM. x axis will be age
# fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# # Plot 1: volume vs age
# ax[0, 0].scatter(age, volume_total_standardized, c=colors)
# ax[0, 0].set_xlabel('Age')
# ax[0, 0].set_ylabel('Volume Total (Standardized)')
# correlation1, p_value1 = pearsonr(age, volume_total_standardized)
# ax[0, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0, 0].transAxes,
#               ha='center')
# # Plot 2: radial diffusivity vs age
# ax[0, 1].scatter(age, radial_diffusivity_standardized, c=colors)
# ax[0, 1].set_xlabel('Age')
# ax[0, 1].set_ylabel('Radial Diffusivity (Standardized)')
# correlation2, p_value2 = pearsonr(age, radial_diffusivity_standardized)
# ax[0, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[0, 1].transAxes,
#               ha='center')
# # Plot 3: x vs age
# ax[0, 2].scatter(age, x, c=colors)
# ax[0, 2].set_xlabel('Age')
# ax[0, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
# correlation3, p_value3 = pearsonr(age, x)
# ax[0, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[0, 2].transAxes,
#               ha='center')
# # Plot 4: torque0 vs age
# ax[1, 0].scatter(age, torque0, c=colors)
# ax[1, 0].set_xlabel('Age')
# ax[1, 0].set_ylabel('Isometric Torque')
# correlation4, p_value4 = pearsonr(age, torque0)
# ax[1, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[1, 0].transAxes,
#               ha='center')
# # Plot 5: torque4 vs age
# ax[1, 1].scatter(age, torque4, c=colors)
# ax[1, 1].set_xlabel('Age')
# ax[1, 1].set_ylabel('Torque (120 deg/s)')
# correlation5, p_value5 = pearsonr(age, torque4)
# ax[1, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[1, 1].transAxes,
#               ha='center')
# # Plot 6: KEM vs age
# ax[1, 2].scatter(age, peak_kem_stand, c=colors)
# ax[1, 2].set_xlabel('Age')
# ax[1, 2].set_ylabel('Peak KEM Stand')
# correlation6, p_value6 = pearsonr(age, peak_kem_stand)
# ax[1, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation6, p_value6), transform=ax[1, 2].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'finalFigures/age_vs_all_timepoint1_bySex.svg'),
#             format='svg', dpi=300)
# plt.show()


df = pd.DataFrame({
    'age': age,
    'sex': [1 if s == 'F' else 0 for s in sex_vector],  # Convert to binary (0 = M, 1 = F)
    'volume': volume_total_standardized,
    'radial_diffusivity': radial_diffusivity_standardized,
    'composite_x': x,
    'isometric': torque0 / weight_height * 100,  # Convert to percentage of weight-height
    'isokinetic': torque4 / weight_height * 100,  # Convert to percentage of weight-height
    'kem': peak_kem_stand / weight_height * 100  # Convert to percentage of weight-height
    # 'isometric': torque0 / volume_total,
    # 'isokinetic': torque4 / volume_total,
    # 'kem': peak_kem_stand / volume_total
})

print()
outcomes = ['volume', 'radial_diffusivity', 'composite_x', 'isometric', 'isokinetic', 'kem']
for outcome in outcomes:
    y = df[outcome]
    X = sm.add_constant(df[['age', 'sex']])  # Add constant for intercept
    model = sm.OLS(y, X).fit()
    print(f"Outcome: {outcome}")
    print(model.summary())

df['sex_label'] = ['M' if s ==0 else 'F' for s in df['sex']]
outcome_labels = ['Volume (Standardized)', 'Radial Diffusivity (Standardized)',
                  'Radial Diffusivity + Volume Total (Standardized)',
                    'Isometric Torque (%BW*h)', '120 deg/s Isokinetic Torque (%BW*h)', 'Peak KEM Stand (%BW*h)']
                    # 'Specific Isometric Torque (N*m/cm³)', '120 deg/s Specific Isokinetic Torque (N*m/cm³)', 'Specific Peak KEM Stand (N*m/cm³)']

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()
for i, (outcome, outcome_label) in enumerate(zip(outcomes, outcome_labels)):
    # Sex-specific regression lines
    for sex_val, color in zip(['M', 'F'], ['blue', 'orange']):
        subset = df[df['sex_label'] == sex_val]
        sns.regplot(
            data=subset, x='age', y=outcome, ax=ax[i],
            scatter=True,
            scatter_kws={'s': 50, 'alpha': 0.5},
            label=sex_val,
            ci=None,
            color=color
        )

    # Multiple regression: age and sex → outcome
    y = df[outcome]
    X = sm.add_constant(df[['age', 'sex']])
    model = sm.OLS(y, X).fit()

    # Extract coefficients and p-values
    age_coef = model.params['age']
    age_p = model.pvalues['age']
    sex_coef = model.params['sex']
    sex_p = model.pvalues['sex']

    # Annotate with both predictors
    ax[i].text(0.5, 0.93, f"Age:   β = {age_coef:.2f}, p = {age_p:.2e}",
               transform=ax[i].transAxes, ha='center', fontsize=10)
    ax[i].text(0.5, 0.86, f"Sex:   β = {sex_coef:.2f}, p = {sex_p:.2e}",
               transform=ax[i].transAxes, ha='center', fontsize=10)

    ax[i].set_xlabel('Age')
    ax[i].set_ylabel(outcome_label)

# Shared legend
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Sex', loc='upper right', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Age and Sex Effects on MRI Metrics and Kinetics', fontsize=16)

# Save
plt.savefig(os.path.join(repo_dir, 'finalFigures/age_vs_all_timepoint1_bySex_regression_byBwH.svg'),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()



