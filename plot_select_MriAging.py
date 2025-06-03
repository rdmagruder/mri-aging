import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import re

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


# FIGURE 1
def plot_correls_1x4(y, ylabel, save_name):

    # Make a subplot of 4 plots. All with input y on the y axis. the x axis will be kem_stand, sts_time, torso_orientation_liftoff, peak_torso_ang_vel
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot 1: y vs Peak KEM Stand
    ax[0].scatter(peak_kem_stand, y, c=colors)
    ax[0].set_xlabel('Peak KEM Stand')
    ax[0].set_ylabel(ylabel)
    correlation1, p_value1 = pearsonr(peak_kem_stand, y)
    ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
    # Plot 2: y vs STS Time
    ax[1].scatter(STS_time, y, c=colors)
    ax[1].set_xlabel('STS Time')
    ax[1].set_ylabel(ylabel)
    correlation2, p_value2 = pearsonr(STS_time, y)
    ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
    # Plot 3: y vs Torso Orientation at Liftoff
    ax[2].scatter(torso_orientation_liftoff, y, c=colors)
    ax[2].set_xlabel('Torso Orientation at Liftoff')
    ax[2].set_ylabel(ylabel)
    correlation3, p_value3 = pearsonr(torso_orientation_liftoff, y)
    ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
    # Plot 4: y vs Peak Torso Ang Velocity
    ax[3].scatter(peak_torso_ang_vel, y, c=colors)
    ax[3].set_xlabel('Peak Torso Ang Velocity')
    ax[3].set_ylabel(ylabel)
    correlation4, p_value4 = pearsonr(peak_torso_ang_vel, y)
    ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
    # Save the figure as an svg file with the name save_name
    plt.savefig(os.path.join(repo_dir, save_name), format='svg', dpi=300)
    plt.show()

plot_correls_1x4(x, 'Radial Diffusivity * Volume Total', 'trimmedFigures/RDxVol_timepoint{}_bySex.svg'.format(timepoint_to_plot))

# 3x3 subplot. The y axis for each column will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). the x axis will be the same for each row. The rows will be torque0, torque4, and KEM
# Make a subplot of 3x3 plots. The y axis for each column will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). the x axis will be the same for each row. The rows will be torque0, torque4, and KEM
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
# Plot 1: volume vs torque0
ax[0, 0].scatter(torque0, volume_total_standardized, c=colors)
ax[0, 0].set_xlabel('Torque 0')
ax[0, 0].set_ylabel('Volume Total (Standardized)')
correlation1, p_value1 = pearsonr(torque0, volume_total_standardized)
ax[0, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0, 0].transAxes,
              ha='center')
# Plot 2: radial diffusivity vs torque0
ax[0, 1].scatter(torque0, radial_diffusivity_standardized, c=colors)
ax[0, 1].set_xlabel('Torque 0')
ax[0, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation2, p_value2 = pearsonr(torque0, radial_diffusivity_standardized)
ax[0, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[0, 1].transAxes,
              ha='center')
# Plot 3: x vs torque0
ax[0, 2].scatter(torque0, x, c=colors)
ax[0, 2].set_xlabel('Torque 0')
ax[0, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation3, p_value3 = pearsonr(torque0, x)
ax[0, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[0, 2].transAxes,
              ha='center')
# Plot 4: volume vs torque4
ax[1, 0].scatter(torque4, volume_total_standardized, c=colors)
ax[1, 0].set_xlabel('Torque 4')
ax[1, 0].set_ylabel('Volume Total (Standardized)')
correlation4, p_value4 = pearsonr(torque4, volume_total_standardized)
ax[1, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[1, 0].transAxes,
              ha='center')
# Plot 5: radial diffusivity vs torque4
ax[1, 1].scatter(torque4, radial_diffusivity_standardized, c=colors)
ax[1, 1].set_xlabel('Torque 4')
ax[1, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation5, p_value5 = pearsonr(torque4, radial_diffusivity_standardized)
ax[1, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[1, 1].transAxes,
              ha='center')
# Plot 6: x vs torque4
ax[1, 2].scatter(torque4, x, c=colors)
ax[1, 2].set_xlabel('Torque 4')
ax[1, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation6, p_value6 = pearsonr(torque4, x)
ax[1, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation6, p_value6), transform=ax[1, 2].transAxes,
              ha='center')
# Plot 7: volume vs peak_kem_stand
ax[2, 0].scatter(peak_kem_stand, volume_total_standardized, c=colors)
ax[2, 0].set_xlabel('Peak KEM Stand')
ax[2, 0].set_ylabel('Volume Total (Standardized)')
correlation7, p_value7 = pearsonr(peak_kem_stand, volume_total_standardized)
ax[2, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation7, p_value7), transform=ax[2, 0].transAxes,
              ha='center')
# Plot 8: radial diffusivity vs peak_kem_stand
ax[2, 1].scatter(peak_kem_stand, radial_diffusivity_standardized, c=colors)
ax[2, 1].set_xlabel('Peak KEM Stand')
ax[2, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation8, p_value8 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
ax[2, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation8, p_value8), transform=ax[2, 1].transAxes,
              ha='center')
# Plot 9: x vs peak_kem_stand
ax[2, 2].scatter(peak_kem_stand, x, c=colors)
ax[2, 2].set_xlabel('Peak KEM Stand')
ax[2, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation9, p_value9 = pearsonr(peak_kem_stand, x)
ax[2, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation9, p_value9), transform=ax[2, 2].transAxes,
              ha='center')
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'trimmedFigures/torqueAndKEM_vs_MRI_timepoint{}_bySex.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300)
plt.show()

isokin_by_isomet = torque4/torque0
KEM_by_isomet = peak_kem_stand/torque0
# make a 2x3 subplot. y axis will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). row 1 will be isokin_by_isomet, row 2 will be KEM_by_isomet
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# Plot 1: volume vs isokin_by_isomet
ax[0, 0].scatter(isokin_by_isomet, volume_total_standardized, c=colors)
ax[0, 0].set_xlabel('Isokinetic / Isometric Torque')
ax[0, 0].set_ylabel('Volume Total (Standardized)')
correlation1, p_value1 = pearsonr(isokin_by_isomet, volume_total_standardized)
ax[0, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0, 0].transAxes,
              ha='center')
# Plot 2: radial diffusivity vs isokin_by_isomet
ax[0, 1].scatter(isokin_by_isomet, radial_diffusivity_standardized, c=colors)
ax[0, 1].set_xlabel('Isokinetic / Isometric Torque')
ax[0, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation2, p_value2 = pearsonr(isokin_by_isomet, radial_diffusivity_standardized)
ax[0, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[0, 1].transAxes,
              ha='center')
# Plot 3: x vs isokin_by_isomet
ax[0, 2].scatter(isokin_by_isomet, x, c=colors)
ax[0, 2].set_xlabel('Isokinetic / Isometric Torque')
ax[0, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation3, p_value3 = pearsonr(isokin_by_isomet, x)
ax[0, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[0, 2].transAxes,
              ha='center')
# Plot 4: volume vs KEM_by_isomet
ax[1, 0].scatter(KEM_by_isomet, volume_total_standardized, c=colors)
ax[1, 0].set_xlabel('KEM / Isometric Torque')
ax[1, 0].set_ylabel('Volume Total (Standardized)')
correlation4, p_value4 = pearsonr(KEM_by_isomet, volume_total_standardized)
ax[1, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[1, 0].transAxes,
              ha='center')
# Plot 5: radial diffusivity vs KEM_by_isomet
ax[1, 1].scatter(KEM_by_isomet, radial_diffusivity_standardized, c=colors)
ax[1, 1].set_xlabel('KEM / Isometric Torque')
ax[1, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation5, p_value5 = pearsonr(KEM_by_isomet, radial_diffusivity_standardized)
ax[1, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[1, 1].transAxes,
              ha='center')
# Plot 6: x vs KEM_by_isomet
ax[1, 2].scatter(KEM_by_isomet, x, c=colors)
ax[1, 2].set_xlabel('KEM / Isometric Torque')
ax[1, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation6, p_value6 = pearsonr(KEM_by_isomet, x)
ax[1, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation6, p_value6), transform=ax[1, 2].transAxes,
              ha='center')
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isokinAndKEM_by_isomet_vs_MRI_timepoint{}_bySex.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300)
plt.show()

# # make a 1x4 subplot. y axis will be torque0, torque4, KEM, and x (combined radial diffusivity and volume). the x axis will be age
# fig, ax = plt.subplots(1, 4, figsize=(20, 5))
# # Plot 1: torque0 vs age
# ax[0].scatter(age, torque0)
# ax[0].set_xlabel('Age')
# ax[0].set_ylabel('Isometric Torque')
# correlation1, p_value1 = pearsonr(age, torque0)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#               ha='center')
# # Plot 2: torque4 vs age
# ax[1].scatter(age, torque4)
# ax[1].set_xlabel('Age')
# ax[1].set_ylabel('Torque (120 deg/s)')
# correlation2, p_value2 = pearsonr(age, torque4)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#               ha='center')
# # Plot 3: KEM vs age
# ax[2].scatter(age, peak_kem_stand)
# ax[2].set_xlabel('Age')
# ax[2].set_ylabel('Peak KEM Stand')
# correlation3, p_value3 = pearsonr(age, peak_kem_stand)
# ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes,
#               ha='center')
# # Plot 4: x vs age
# ax[3].scatter(age, x)
# ax[3].set_xlabel('Age')
# ax[3].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
# correlation4, p_value4 = pearsonr(age, x)
# ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'trimmedFigures/age_vs_various_timepoint{}.svg'.format(timepoint_to_plot)),
#             format='svg', dpi=300)
# plt.show()

# make a 2x3 subplot. y axis row 1 will be standardized volume, radial diffusivity, and x (combined radial diffusivity and volume). row 2 will be torque0, torque4, and KEM. x axis will be age
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
# Plot 1: volume vs age
ax[0, 0].scatter(age, volume_total_standardized, c=colors)
ax[0, 0].set_xlabel('Age')
ax[0, 0].set_ylabel('Volume Total (Standardized)')
correlation1, p_value1 = pearsonr(age, volume_total_standardized)
ax[0, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0, 0].transAxes,
              ha='center')
# Plot 2: radial diffusivity vs age
ax[0, 1].scatter(age, radial_diffusivity_standardized, c=colors)
ax[0, 1].set_xlabel('Age')
ax[0, 1].set_ylabel('Radial Diffusivity (Standardized)')
correlation2, p_value2 = pearsonr(age, radial_diffusivity_standardized)
ax[0, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[0, 1].transAxes,
              ha='center')
# Plot 3: x vs age
ax[0, 2].scatter(age, x, c=colors)
ax[0, 2].set_xlabel('Age')
ax[0, 2].set_ylabel('Radial Diffusivity * Volume Total (Standardized)')
correlation3, p_value3 = pearsonr(age, x)
ax[0, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[0, 2].transAxes,
              ha='center')
# Plot 4: torque0 vs age
ax[1, 0].scatter(age, torque0, c=colors)
ax[1, 0].set_xlabel('Age')
ax[1, 0].set_ylabel('Isometric Torque')
correlation4, p_value4 = pearsonr(age, torque0)
ax[1, 0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[1, 0].transAxes,
              ha='center')
# Plot 5: torque4 vs age
ax[1, 1].scatter(age, torque4, c=colors)
ax[1, 1].set_xlabel('Age')
ax[1, 1].set_ylabel('Torque (120 deg/s)')
correlation5, p_value5 = pearsonr(age, torque4)
ax[1, 1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[1, 1].transAxes,
              ha='center')
# Plot 6: KEM vs age
ax[1, 2].scatter(age, peak_kem_stand, c=colors)
ax[1, 2].set_xlabel('Age')
ax[1, 2].set_ylabel('Peak KEM Stand')
correlation6, p_value6 = pearsonr(age, peak_kem_stand)
ax[1, 2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation6, p_value6), transform=ax[1, 2].transAxes,
              ha='center')
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'trimmedFigures/age_vs_all_timepoint1_bySex.svg'),
            format='svg', dpi=300)
plt.show()

#
# isokin_by_vol = torque4/volume_total_standardized
# KEM_by_vol = peak_kem_stand/volume_total_standardized
# # make a 1x2 subplot. y axis will be radial diffusivity. x axis will be isokin_by_vol, KEM_by_vol
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # Plot 1: radial diffusivity vs isokin_by_vol
# ax[0].scatter(isokin_by_vol, radial_diffusivity_standardized, c=colors)
# ax[0].set_xlabel('Isokinetic / Volume Total')
# ax[0].set_ylabel('Radial Diffusivity (Standardized)')
# correlation1, p_value1 = pearsonr(isokin_by_vol, radial_diffusivity_standardized)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#               ha='center')
# # Plot 2: radial diffusivity vs KEM_by_vol
# ax[1].scatter(KEM_by_vol, radial_diffusivity_standardized, c=colors)
# ax[1].set_xlabel('KEM / Volume Total')
# ax[1].set_ylabel('Radial Diffusivity (Standardized)')
# correlation2, p_value2 = pearsonr(KEM_by_vol, radial_diffusivity_standardized)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isokinAndKEM_by_vol_vs_radialDiffusivity_timepoint1_bySex.svg'),
#             format='svg', dpi=300)
# plt.show()
#
# isokin_by_x = torque4/x
# KEM_by_x = peak_kem_stand/x
# # make a 1x2 subplot. first plot will be torque4 vs peak_kem_stand, second plot will be isokin_by_x vs KEM_by_x
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # Plot 1: torque4 vs peak_kem_stand
# ax[0].scatter(torque4, peak_kem_stand, c=colors)
# ax[0].set_xlabel('Torque (120 deg/s)')
# ax[0].set_ylabel('Peak KEM Stand')
# correlation1, p_value1 = pearsonr(torque4, peak_kem_stand)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#               ha='center')
# # Plot 2: isokin_by_x vs KEM_by_x
# ax[1].scatter(isokin_by_x, KEM_by_x, c=colors)
# ax[1].set_xlabel('Isokinetic / Radial Diffusivity * Volume Total')
# ax[1].set_ylabel('KEM / Radial Diffusivity * Volume Total')
# correlation2, p_value2 = pearsonr(isokin_by_x, KEM_by_x)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isokinAndKEM_by_x_vs_radialDiffusivity_timepoint1_bySex.svg'),
#             format='svg', dpi=300)
# plt.show()
#
# isomet_by_vol = torque0/volume_total_standardized
# KEM_by_vol = peak_kem_stand/volume_total_standardized
# # make a 1x2 subplot. y axis will be radial diffusivity. x axis will be isomet_by_vol, KEM_by_vol
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # Plot 1: radial diffusivity vs isomet_by_vol
# ax[0].scatter(isomet_by_vol, radial_diffusivity_standardized, c=colors)
# ax[0].set_xlabel('Isometric / Volume Total')
# ax[0].set_ylabel('Radial Diffusivity (Standardized)')
# correlation1, p_value1 = pearsonr(isomet_by_vol, radial_diffusivity_standardized)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#                 ha='center')
# # Plot 2: radial diffusivity vs KEM_by_vol
# ax[1].scatter(KEM_by_vol, radial_diffusivity_standardized, c=colors)
# ax[1].set_xlabel('KEM / Volume Total')
# ax[1].set_ylabel('Radial Diffusivity (Standardized)')
# correlation2, p_value2 = pearsonr(KEM_by_vol, radial_diffusivity_standardized)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#                 ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isometAndKEM_by_vol_vs_radialDiffusivity_timepoint1_bySex.svg'),
#             format='svg', dpi=300)
# plt.show()
#
# isomet_by_x = torque0/x
# KEM_by_x = peak_kem_stand/x
# # make a 1x2 subplot. first plot will be torque0 vs peak_kem_stand, second plot will be isomet_by_x vs KEM_by_x
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # Plot 1: torque0 vs peak_kem_stand
# ax[0].scatter(torque0, peak_kem_stand, c=colors)
# ax[0].set_xlabel('Torque (Isometric)')
# ax[0].set_ylabel('Peak KEM Stand')
# correlation1, p_value1 = pearsonr(torque0, peak_kem_stand)
# ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
#               ha='center')
# # Plot 2: isomet_by_x vs KEM_by_x
# ax[1].scatter(isomet_by_x, KEM_by_x, c=colors)
# ax[1].set_xlabel('Isometric / Radial Diffusivity * Volume Total')
# ax[1].set_ylabel('KEM / Radial Diffusivity * Volume Total')
# correlation2, p_value2 = pearsonr(isomet_by_x, KEM_by_x)
# ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
#               ha='center')
# # Save the figure as an svg file with the name save_name
# plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isometAndKEM_by_x_vs_radialDiffusivity_timepoint1_bySex.svg'),
#             format='svg', dpi=300)
# plt.show()
#


# Multiple regression to predict MRI-derived x from both torque4 and KEM

# Standardize the predictors
torque4_std = (torque4 - np.mean(torque4)) / np.std(torque4)
kem_std = (peak_kem_stand - np.mean(peak_kem_stand)) / np.std(peak_kem_stand)

# Create design matrix
X_mri = np.column_stack((torque4_std, kem_std))
X_mri = sm.add_constant(X_mri)

# Fit the model
model_mri = sm.OLS(x, X_mri)
results_mri = model_mri.fit()

# Print summary
print(results_mri.summary())

# Predict x from model
x_pred = results_mri.predict(X_mri)

# make a 1x2 subplot. first plot will be isokinetic torque vs KEM, second plot will be x vs x_pred
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Plot 1: isokinetic torque vs KEM
ax[0].scatter(torque4, peak_kem_stand, c=colors)
ax[0].set_xlabel('Isokinetic Torque (120 deg/s)')
ax[0].set_ylabel('Peak KEM Stand')
correlation1, p_value1 = pearsonr(torque4, peak_kem_stand)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes,
              ha='center')
# Plot 2: x vs x_pred
ax[1].scatter(x, x_pred, c=colors)
ax[1].set_xlabel('Radial Diffusivity * Volume Total (Standardized)')
ax[1].set_ylabel('Predicted Radial Diffusivity * Volume Total (Standardized)')
correlation2, p_value2 = pearsonr(x, x_pred)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes,
              ha='center')
# Save the figure as an svg file with the name save_name
plt.savefig(os.path.join(repo_dir, 'trimmedFigures/isokinAndKEM_vs_MRI_pred_timepoint{}_bySex.svg'.format(timepoint_to_plot)),
            format='svg', dpi=300)
plt.show()

