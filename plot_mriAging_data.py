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
            data_dict[muscle]['Radial Diffusivity'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Radial_Diffusivity'].mean()
            data_dict[muscle]['Volume'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Volume'].mean()
            data_dict[muscle]['Fat Fraction'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Fat_Fraction'].mean()
            data_dict[muscle]['CSA'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['CSA'].mean()
            for k, velocity in enumerate(velocities):
                data_dict[muscle]['Torque'][i, j, k] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle) & (data['Velocity'] == velocity)]['Torque'].mean()

            # get the peak_kem_stand columns and average for this subject and timepoint - average over whole matrix
            data_dict['Peak KEM Stand'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_kem_stand_columns])
            data_dict['Peak KEM Sit'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_kem_sit_columns])
            data_dict['Peak HFM Stand'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_hfm_stand_columns])
            data_dict['Peak HFM Sit'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_hfm_sit_columns])
            data_dict['Peak Force Sit'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_force_sit_columns])
            data_dict['Peak Force Stand'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_force_stand_columns])
            data_dict['Peak Torso Ang Velocity'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_torso_ang_vel_columns])
            data_dict['Torso Orientation at Liftoff'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in torso_orientation_liftoff_columns])

            # Sum the volumes for each muscle
            if muscle == muscles[-1] :
                data_dict['Volume total'][i, j] = np.sum(data_dict[muscle]['Volume'][i, j] for muscle in muscles)
                data_dict['contractile_volume'][i, j] = np.sum([data_dict[muscle]['Volume'][i, j] * (1- data_dict[muscle]['Fat Fraction'][i, j]/100) for muscle in muscles])

                # get the age for this subject and timepoint
                data_dict['Age'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Age'].values[0]
                # get the STS_time for this subject and timepoint
                data_dict['STS_time'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['STS_stopwatch'].values[0]
            
# for each subject and timepoint, create a weighted average across muscles weighted by volume
# create a dictionary to store the weighted average for each subject and timepoint
weighted_diffusivity = np.zeros((len(subjects), len(timepoints)))
weighted_fat_fraction = np.zeros((len(subjects), len(timepoints)))
for i, subject in enumerate(subjects):
    for j, timepoint in enumerate(timepoints):
        total_volume = np.sum([data_dict[muscle]['Volume'][i, j] for muscle in muscles])
        weighted_diffusivity[i, j] = np.sum([data_dict[muscle]['Radial Diffusivity'][i, j] * data_dict[muscle]['Volume'][i, j] / total_volume for muscle in muscles])
        weighted_fat_fraction[i, j] = np.sum([data_dict[muscle]['Fat Fraction'][i, j] * data_dict[muscle]['Volume'][i, j] / total_volume for muscle in muscles])

# Second timepoint
timepoint_to_plot = 1
velocity_to_plot = 0 #0, 1, 2, 3, 4 corresponds to 0, -60, -45, 90, 120
remove_nans = True

radial_diffusivity = weighted_diffusivity[:,timepoint_to_plot]
fat_fraction = weighted_fat_fraction[:,timepoint_to_plot]
volume_total = data_dict['Volume total'][:,timepoint_to_plot]
contractile_volume = data_dict['contractile_volume'][:,timepoint_to_plot]
peak_kem_stand = data_dict['Peak KEM Stand'][:,timepoint_to_plot]
peak_kem_sit = data_dict['Peak KEM Sit'][:,timepoint_to_plot]
peak_hfm_stand = data_dict['Peak HFM Stand'][:,timepoint_to_plot]
peak_hfm_sit = data_dict['Peak HFM Sit'][:,timepoint_to_plot]
peak_force_stand = data_dict['Peak Force Stand'][:,timepoint_to_plot]
peak_force_sit = data_dict['Peak Force Sit'][:,timepoint_to_plot]
peak_torso_ang_vel = data_dict['Peak Torso Ang Velocity'][:,timepoint_to_plot]
torso_orientation_liftoff = data_dict['Torso Orientation at Liftoff'][:,timepoint_to_plot]
age = data_dict['Age'][:,timepoint_to_plot]
STS_time = data_dict['STS_time'][:,timepoint_to_plot]
torque0 = data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,0]
torque1 = data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,1]
torque2 = data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,2]
torque3 = data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,3]
torque4 = data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,4]

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


# multiple regression model to predict peak_kem_stand from some number of inputs
import statsmodels.api as sm
# create a design matrix
radial_diffusivity_standardized = (radial_diffusivity - np.mean(radial_diffusivity)) / np.std(radial_diffusivity)
volume_total_standardized = (volume_total - np.mean(volume_total)) / np.std(volume_total)
contractile_volume_standardized = (contractile_volume - np.mean(contractile_volume)) / np.std(contractile_volume)

# X = np.column_stack((torque0, torso_orientation_liftoff, peak_torso_ang_vel)) # only torque is sig
X = np.column_stack((torque0, torque1, torque2, torque3, torque4))
X = sm.add_constant(X)
# fit the model
model = sm.OLS(peak_kem_stand, X)
results = model.fit()
# print the results
print(results.summary())


# compute correlation between peak_kem_stand and radial_diffusivity
# x = radial_diffusivity_standardized + volume_total_standardized
# x = radial_diffusivity_standardized + contractile_volume_standardized
x = radial_diffusivity_standardized + volume_total_standardized

# Define a function to make the 2 sets of subplots
def plot_correls(y, ylabel, save_name1, save_name2):
    # Make a subplot of 4 plots. All with input y on the y axis. the x axis will be kem_stand, kem_sit, hfm_stand, hfm_sit
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot 1: y vs Peak KEM Stand
    ax[0].scatter(peak_kem_stand, y)
    ax[0].set_xlabel('Peak KEM Stand')
    ax[0].set_ylabel(ylabel)
    correlation1, p_value1 = pearsonr(peak_kem_stand, y)
    ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
    # Plot 2: y vs Peak KEM Sit
    ax[1].scatter(peak_kem_sit, y)
    ax[1].set_xlabel('Peak KEM Sit')
    ax[1].set_ylabel(ylabel)
    correlation2, p_value2 = pearsonr(peak_kem_sit, y)
    ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
    # Plot 3: y vs Peak HFM Stand
    ax[2].scatter(peak_hfm_stand, y)
    ax[2].set_xlabel('Peak HFM Stand')
    ax[2].set_ylabel(ylabel)
    correlation3, p_value3 = pearsonr(peak_hfm_stand, y)
    ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
    # Plot 4: y vs Peak HFM Sit
    ax[3].scatter(peak_hfm_sit, y)
    ax[3].set_xlabel('Peak HFM Sit')
    ax[3].set_ylabel(ylabel)
    correlation4, p_value4 = pearsonr(peak_hfm_sit, y)
    ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
    # Save the figure as an svg file with the name save_name1
    plt.savefig(os.path.join(repo_dir, save_name1), format='svg', dpi=300)
    plt.show()

    # Make a subplot of 3 plots. All with input y on the y axis. the x axis will be STS time, torso orientation at liftoff, peak torso ang vel
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Plot 1: y vs STS Time
    ax[0].scatter(STS_time, y)
    ax[0].set_xlabel('STS Time')
    ax[0].set_ylabel(ylabel)
    correlation5, p_value5 = pearsonr(STS_time, y)
    ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[0].transAxes, ha='center')
    # Plot 2: y vs Torso Orientation at Liftoff
    ax[1].scatter(torso_orientation_liftoff, y)
    ax[1].set_xlabel('Torso Orientation at Liftoff')
    ax[1].set_ylabel(ylabel)
    correlation6, p_value6 = pearsonr(torso_orientation_liftoff, y)
    ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation6, p_value6), transform=ax[1].transAxes, ha='center')
    # Plot 3: y vs Peak Torso Ang Velocity
    ax[2].scatter(peak_torso_ang_vel, y)
    ax[2].set_xlabel('Peak Torso Ang Velocity')
    ax[2].set_ylabel(ylabel)
    correlation7, p_value7 = pearsonr(peak_torso_ang_vel, y)
    ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation7, p_value7), transform=ax[2].transAxes, ha='center')
    # Save the figure as an svg file with the name save_name2
    plt.savefig(os.path.join(repo_dir, save_name2), format='svg', dpi=300)
    plt.show()

# Plot correls for radial diffusivity standardized
plot_correls(radial_diffusivity_standardized, 'Radial Diffusivity Standardized', 'RD_vs_moments_timepoint{}.svg'.format(timepoint_to_plot), 'RD_vs_STSmetrics_timepoint{}.svg'.format(timepoint_to_plot))

# Plot correls for volume total standardized
plot_correls(volume_total_standardized, 'Volume Total Standardized', 'Vol_Total_vs_moments_timepoint{}.svg'.format(timepoint_to_plot), 'Vol_Total_vs_STSmetrics_timepoint{}.svg'.format(timepoint_to_plot))

# Plot correls for contractile volume standardized
plot_correls(contractile_volume_standardized, 'Contractile Volume Standardized', 'CV_vs_moments_timepoint{}.svg'.format(timepoint_to_plot), 'CV_vs_STSmetrics_timepoint{}.svg'.format(timepoint_to_plot))

# Plot correls for fat fraction
plot_correls(fat_fraction, 'Fat Fraction', 'FF_vs_moments_timepoint{}.svg'.format(timepoint_to_plot), 'FF_vs_STSmetrics_timepoint{}.svg'.format(timepoint_to_plot))

# Plot correls for Radial Diffusivity * Volume Total
plot_correls(x, 'Radial Diffusivity * Volume Total', 'RDxVol_vs_moments_timepoint{}.svg'.format(timepoint_to_plot), 'RDxVol_vs_STSmetrics_timepoint{}.svg'.format(timepoint_to_plot))

# Plot correls for all torques
for i, torque in enumerate(torques.T):
    plot_correls(torque, 'Torque {} deg/s'.format(velocities[i]), 'torque_{}deg_vs_moments_timepoint{}.svg'.format(velocities[i], timepoint_to_plot), 'torque_{}deg_vs_STSmetrics_timepoint{}.svg'.format(velocities[i], timepoint_to_plot))

def plot_correls_1x4(y, ylabel, save_name):
    # Make a subplot of 4 plots. All with input y on the y axis. the x axis will be kem_stand, sts_time, torso_orientation_liftoff, peak_torso_ang_vel
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot 1: y vs Peak KEM Stand
    ax[0].scatter(peak_kem_stand, y)
    ax[0].set_xlabel('Peak KEM Stand')
    ax[0].set_ylabel(ylabel)
    correlation1, p_value1 = pearsonr(peak_kem_stand, y)
    ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
    # Plot 2: y vs STS Time
    ax[1].scatter(STS_time, y)
    ax[1].set_xlabel('STS Time')
    ax[1].set_ylabel(ylabel)
    correlation2, p_value2 = pearsonr(STS_time, y)
    ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
    # Plot 3: y vs Torso Orientation at Liftoff
    ax[2].scatter(torso_orientation_liftoff, y)
    ax[2].set_xlabel('Torso Orientation at Liftoff')
    ax[2].set_ylabel(ylabel)
    correlation3, p_value3 = pearsonr(torso_orientation_liftoff, y)
    ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
    # Plot 4: y vs Peak Torso Ang Velocity
    ax[3].scatter(peak_torso_ang_vel, y)
    ax[3].set_xlabel('Peak Torso Ang Velocity')
    ax[3].set_ylabel(ylabel)
    correlation4, p_value4 = pearsonr(peak_torso_ang_vel, y)
    ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
    # Save the figure as an svg file with the name save_name
    plt.savefig(os.path.join(repo_dir, save_name), format='svg', dpi=300)
    plt.show()

# Plot correls for all torques
for i, torque in enumerate(torques.T): #save to PaperFigures folder
    plot_correls_1x4(torque, 'Torque {} deg/s'.format(velocities[i]), 'PaperFigures/torque_{}deg_timepoint{}.svg'.format(velocities[i], timepoint_to_plot))

# plot correls for Radial Diffusivity * Volume Total
plot_correls_1x4(x, 'Radial Diffusivity * Volume Total', 'PaperFigures/RDxVol_timepoint{}.svg'.format(timepoint_to_plot))
# plot correls for Volume Total standardized
plot_correls_1x4(volume_total_standardized, 'Volume Total Standardized', 'PaperFigures/Vol_Total_timepoint{}.svg'.format(timepoint_to_plot))
# plot correls for Contractile Volume standardized
plot_correls_1x4(contractile_volume_standardized, 'Contractile Volume Standardized', 'PaperFigures/CV_timepoint{}.svg'.format(timepoint_to_plot))
# plot correls for Fat Fraction
plot_correls_1x4(fat_fraction, 'Fat Fraction', 'PaperFigures/FF_timepoint{}.svg'.format(timepoint_to_plot))
# plot correls for Radial Diffusivity standardized
plot_correls_1x4(radial_diffusivity_standardized, 'Radial Diffusivity Standardized', 'PaperFigures/RD_timepoint{}.svg'.format(timepoint_to_plot))

# 1x4 subplot. Plot KEM moment on x axis and Total volume standardized, fat fraction, radial diffusivity standardized, and RDxVol standardized on y axis
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
# Plot 1: KEM moment vs Total volume standardized
ax[0].scatter(peak_kem_stand, volume_total_standardized)
ax[0].set_xlabel('Peak KEM Stand')
ax[0].set_ylabel('Total Volume Standardized')
correlation1, p_value1 = pearsonr(peak_kem_stand, volume_total_standardized)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
# Plot 2: KEM moment vs Fat Fraction
ax[1].scatter(peak_kem_stand, fat_fraction)
ax[1].set_xlabel('Peak KEM Stand')
ax[1].set_ylabel('Fat Fraction')
correlation2, p_value2 = pearsonr(peak_kem_stand, fat_fraction)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
# Plot 3: KEM moment vs contractile volume standardized
ax[2].scatter(peak_kem_stand, contractile_volume_standardized)
ax[2].set_xlabel('Peak KEM Stand')
ax[2].set_ylabel('Contractile Volume Standardized')
correlation3, p_value3 = pearsonr(peak_kem_stand, contractile_volume_standardized)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
# Plot 4: KEM moment vs Radial Diffusivity Standardized
ax[3].scatter(peak_kem_stand, radial_diffusivity_standardized)
ax[3].set_xlabel('Peak KEM Stand')
ax[3].set_ylabel('Radial Diffusivity Standardized')
correlation4, p_value4 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
# Plot 5: KEM moment vs Radial Diffusivity * Volume Total Standardized
ax[4].scatter(peak_kem_stand, x)
ax[4].set_xlabel('Peak KEM Stand')
ax[4].set_ylabel('Radial Diffusivity * Volume Total Standardized')
correlation5, p_value5 = pearsonr(peak_kem_stand, x)
ax[4].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[4].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'KEM_moments_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()

# 1x4 subplot. Plot KEM moment on x axis and Total volume standardized, fat fraction, radial diffusivity standardized, and RDxVol standardized on y axis
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
# Plot 1: KEM moment vs Total volume standardized
ax[0].scatter(peak_kem_stand, volume_total_standardized)
ax[0].set_xlabel('Peak KEM Stand')
ax[0].set_ylabel('Total Volume Standardized')
correlation1, p_value1 = pearsonr(peak_kem_stand, volume_total_standardized)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
# Plot 3: KEM moment vs contractile volume standardized
ax[1].scatter(peak_kem_stand, contractile_volume_standardized)
ax[1].set_xlabel('Peak KEM Stand')
ax[1].set_ylabel('Contractile Volume Standardized')
correlation3, p_value3 = pearsonr(peak_kem_stand, contractile_volume_standardized)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[1].transAxes, ha='center')
# Plot 4: KEM moment vs Radial Diffusivity Standardized
ax[2].scatter(peak_kem_stand, radial_diffusivity_standardized)
ax[2].set_xlabel('Peak KEM Stand')
ax[2].set_ylabel('Radial Diffusivity Standardized')
correlation4, p_value4 = pearsonr(peak_kem_stand, radial_diffusivity_standardized)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[2].transAxes, ha='center')
# Plot 5: KEM moment vs Radial Diffusivity * Volume Total Standardized
ax[3].scatter(peak_kem_stand, x)
ax[3].set_xlabel('Peak KEM Stand')
ax[3].set_ylabel('Radial Diffusivity * Volume Total Standardized')
correlation5, p_value5 = pearsonr(peak_kem_stand, x)
ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation5, p_value5), transform=ax[3].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'PaperFigures/KEM_moments_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()

KEM_by_torque = peak_kem_stand / torque0
# 1x4 subplot. Plot kem_by_torque on x axis and Total volume standardized, contractile volume standardized, radial diffusivity standardized, and RDxVol standardized on y axis
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
# Plot 1: KEM by Torque vs Total volume standardized
ax[0].scatter(KEM_by_torque, volume_total_standardized)
ax[0].set_xlabel('KEM by Isometric Torque')
ax[0].set_ylabel('Total Volume Standardized')
correlation1, p_value1 = pearsonr(KEM_by_torque, volume_total_standardized)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
# Plot 2: KEM by Torque vs Contractile Volume Standardized
ax[1].scatter(KEM_by_torque, contractile_volume_standardized)
ax[1].set_xlabel('KEM by Isometric Torque')
ax[1].set_ylabel('Contractile Volume Standardized')
correlation2, p_value2 = pearsonr(KEM_by_torque, contractile_volume_standardized)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
# Plot 3: KEM by Torque vs Radial Diffusivity Standardized
ax[2].scatter(KEM_by_torque, radial_diffusivity_standardized)
ax[2].set_xlabel('KEM by Isometric Torque')
ax[2].set_ylabel('Radial Diffusivity Standardized')
correlation3, p_value3 = pearsonr(KEM_by_torque, radial_diffusivity_standardized)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
# Plot 4: KEM by Torque vs Radial Diffusivity * Volume Total Standardized
ax[3].scatter(KEM_by_torque, x)
ax[3].set_xlabel('KEM by Isometric Torque')
ax[3].set_ylabel('Radial Diffusivity * Volume Total Standardized')
correlation4, p_value4 = pearsonr(KEM_by_torque, x)
ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'KEM_by_torque0_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()

KEM_by_torque = peak_kem_stand / torque4
# 1x4 subplot. Plot kem_by_torque on x axis and Total volume standardized, contractile volume standardized, radial diffusivity standardized, and RDxVol standardized on y axis
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
# Plot 1: KEM by Torque vs Total volume standardized
ax[0].scatter(KEM_by_torque, volume_total_standardized)
ax[0].set_xlabel('KEM by Torque 120 deg/s')
ax[0].set_ylabel('Total Volume Standardized')
correlation1, p_value1 = pearsonr(KEM_by_torque, volume_total_standardized)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
# Plot 2: KEM by Torque vs Contractile Volume Standardized
ax[1].scatter(KEM_by_torque, contractile_volume_standardized)
ax[1].set_xlabel('KEM by Torque 120 deg/s')
ax[1].set_ylabel('Contractile Volume Standardized')
correlation2, p_value2 = pearsonr(KEM_by_torque, contractile_volume_standardized)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
# Plot 3: KEM by Torque vs Radial Diffusivity Standardized
ax[2].scatter(KEM_by_torque, radial_diffusivity_standardized)
ax[2].set_xlabel('KEM by Torque 120 deg/s')
ax[2].set_ylabel('Radial Diffusivity Standardized')
correlation3, p_value3 = pearsonr(KEM_by_torque, radial_diffusivity_standardized)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
# Plot 4: KEM by Torque vs Radial Diffusivity * Volume Total Standardized
ax[3].scatter(KEM_by_torque, x)
ax[3].set_xlabel('KEM by Torque 120 deg/s')
ax[3].set_ylabel('Radial Diffusivity * Volume Total Standardized')
correlation4, p_value4 = pearsonr(KEM_by_torque, x)
ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'KEM_by_torque120_vs_MRI_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()

for i, torque in enumerate(torques.T):
    # Make a 1x4 subplot with torque0 on the x and total volume standardized, contractile volume standardized, radial diffusivity standardized, and RDxVol standardized on y axis
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # Plot 1: Torque0 vs Total volume standardized
    ax[0].scatter(torque, volume_total_standardized) #torque with its corresponding velocity
    ax[0].set_xlabel('Torque {} deg/s'.format(velocities[i]))
    ax[0].set_ylabel('Total Volume Standardized')
    correlation1, p_value1 = pearsonr(torque, volume_total_standardized)
    ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
    # Plot 2: Torque0 vs Contractile Volume Standardized
    ax[1].scatter(torque, contractile_volume_standardized)
    ax[1].set_xlabel('Torque {} deg/s'.format(velocities[i]))
    ax[1].set_ylabel('Contractile Volume Standardized')
    correlation2, p_value2 = pearsonr(torque, contractile_volume_standardized)
    ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
    # Plot 3: Torque0 vs Radial Diffusivity Standardized
    ax[2].scatter(torque, radial_diffusivity_standardized)
    ax[2].set_xlabel('Torque {} deg/s'.format(velocities[i]))
    ax[2].set_ylabel('Radial Diffusivity Standardized')
    correlation3, p_value3 = pearsonr(torque, radial_diffusivity_standardized)
    ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3, p_value3), transform=ax[2].transAxes, ha='center')
    # Plot 4: Torque0 vs Radial Diffusivity * Volume Total Standardized
    ax[3].scatter(torque, x)
    ax[3].set_xlabel('Torque {} deg/s'.format(velocities[i]))
    ax[3].set_ylabel('Radial Diffusivity * Volume Total Standardized')
    correlation4, p_value4 = pearsonr(torque, x)
    ax[3].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation4, p_value4), transform=ax[3].transAxes, ha='center')
    plt.savefig(os.path.join(repo_dir, 'PaperFigures/torque_{}deg_vs_MRI_timepoint{}.svg'.format(velocities[i], timepoint_to_plot)), format='svg', dpi=300)
    plt.show()



# Make a 1x5 subplot with torque velocities on the y and peak_kem_stand on the x
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
# Plot 1: Torque vs Peak KEM Stand
for i in range(5):
    ax[i].scatter(peak_kem_stand, data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,i][no_nans])
    ax[i].set_xlabel('Peak KEM Stand')
    # Use velocities to label the y axis
    ax[i].set_ylabel('Torque at {} deg/s'.format(velocities[i]))
    correlation1, p_value1 = pearsonr(peak_kem_stand, data_dict[muscles[0]]['Torque'][:,timepoint_to_plot,i][no_nans])
    ax[i].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[i].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'torque_velocities_vs_kem_stand_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()

# Make a 1x5 subplot with torque velocities on the y and x (radial diffusivity * volume total) on the x
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
# Plot 1: Torque vs radial diffusivity * volume total
for i, torque in enumerate(torques.T):
    ax[i].scatter(x, torque)
    ax[i].set_xlabel('Radial Diffusivity * Volume Total')
    # Use velocities to label the y axis
    ax[i].set_ylabel('Torque at {} deg/s'.format(velocities[i]))
    correlation1, p_value1 = pearsonr(x, torque)
    ax[i].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[i].transAxes, ha='center')
plt.savefig(os.path.join(repo_dir, 'PaperFigures/torque_velocities_vs_RDxVol_timepoint{}.svg'.format(timepoint_to_plot)), format='svg', dpi=300)
plt.show()





# I want to statistically compare the strength of relationship between x and peak_kem_stand and x and STS_time
# I will use a Steiger’s z test to compare the correlation coefficients
from scipy.stats import norm
def steiger_z_test(r1, r2, n):
    # Calculate the z-score
    z = (r1 - r2) / np.sqrt((1 - r1**2) / (n - 2) + (1 - r2**2) / (n - 2))
    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    return z, p_value

# # Calculate the z-score and p-value
# z, p_value = steiger_z_test(correlation3_kem, correlation3_sts, len(x))
# # Print the results
# print(f"Steiger's z-test composite score: z = {z:.2f}, p-value = {p_value:.2e}")
#
# # Calculate the steiger's test for radial diffusivity and volume total
# z, p_value = steiger_z_test(correlation1_kem, correlation1_sts, len(x))
# # Print the results
# #print(f"Steiger's z-test radial diffusivity: z = {z:.2f}, p-value = {p_value:.2e}")
# # Calculate the steiger's test for volume total
# z, p_value = steiger_z_test(correlation2_kem, correlation2_sts, len(x))
# # Print the results
# #print(f"Steiger's z-test volume total: z = {z:.2f}, p-value = {p_value:.2e}")

# # I want predicted x values for kem and STS time, then compute the errors, and compare them using
# # at t-test
# from sklearn.linear_model import LinearRegression
# # create a linear regression model
# model = LinearRegression()
# # fit the model to the data
# model.fit(peak_kem_stand.reshape(-1, 1), x)
# # get the predicted values
# predicted_kem = model.predict(peak_kem_stand.reshape(-1, 1))
# model2 = LinearRegression()
# # fit the model to the data
# model2.fit(STS_time.reshape(-1, 1), x)
# # get the predicted values
# predicted_sts = model2.predict(STS_time.reshape(-1, 1))
# # compute the errors
# error_kem = np.abs(x - predicted_kem)
# error_sts = np.abs(x - predicted_sts)
# print(error_kem)
# print(error_sts)
# # paired samples t-test
# from scipy.stats import ttest_rel
# # Perform the paired t-test
# t_stat, p_value = ttest_rel(error_kem, error_sts)
# # Print the results
# print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}")

