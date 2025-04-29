import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

# repo path is the directory of this file
repo_dir = os.path.dirname(os.path.realpath(__file__))

# Load data
data = pd.read_excel(os.path.join(repo_dir, 'data_all_mh_scott.xlsx'))

# get Radial_Diffusivity and Volume for each muscle, for each Subject and Timepoint
muscles = ['RF', 'VI', 'VM', 'VL']
# define timepoints from data
timepoints = np.unique(data['Timepoint'])
# define subjects from data - it is a string
subjects = pd.unique(data['Subject'])
# remove nan from timepoints and subjects
timepoints = timepoints[~np.isnan(timepoints)]
subjects = subjects[~pd.isnull(subjects)]
# get all columns that start with peak_kem_stand
peak_kem_stand_columns = [col for col in data.columns if col.startswith('peak_kem_stand')]

# create a dictionary to store the data for each muscle
data_dict = {'Peak KEM Stand': np.zeros((len(subjects), len(timepoints))),
             'Volume total': np.zeros((len(subjects), len(timepoints))),
             'contractile_volume': np.zeros((len(subjects), len(timepoints))),
             'Age': np.zeros((len(subjects), len(timepoints))),
             'STS_time': np.zeros((len(subjects), len(timepoints)))}
for muscle in muscles:
    data_dict[muscle] = {'Radial Diffusivity': np.zeros((len(subjects), len(timepoints))),
                         'Volume': np.zeros((len(subjects), len(timepoints))),
                         'Fat Fraction': np.zeros((len(subjects), len(timepoints)))}
    for i, subject in enumerate(subjects):
        for j, timepoint in enumerate(timepoints):
            data_dict[muscle]['Radial Diffusivity'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Radial Diffusivity']
            data_dict[muscle]['Volume'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Volume']
            data_dict[muscle]['Fat Fraction'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint) & (data['Muscle'] == muscle)]['Fat Fraction']
            # get the peak_kem_stand columns and average for this subject and timepoint - average over whole matrix
            data_dict['Peak KEM Stand'][i, j] = np.mean([data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)][col].values[0] for col in peak_kem_stand_columns])
            # Sum the volumes for each muscle
            if muscle == muscles[-1] :
                data_dict['Volume total'][i, j] = np.sum(data_dict[muscle]['Volume'][i, j] for muscle in muscles)
                data_dict['contractile_volume'][i, j] = np.sum([data_dict[muscle]['Volume'][i, j] * (1- data_dict[muscle]['Fat Fraction'][i, j]/100) for muscle in muscles])

                # get the age for this subject and timepoint
                data_dict['Age'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['Age'].values[0]
                # get the STS_time for this subject and timepoint
                data_dict['STS_time'][i, j] = data[(data['Subject'] == subject) & (data['Timepoint'] == timepoint)]['STS_time'].values[0]
            
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
timepoint_to_plot = 1 ; 
radial_diffusivity = weighted_diffusivity[:,timepoint_to_plot]
fat_fraction = weighted_fat_fraction[:,timepoint_to_plot]
volume_total = data_dict['Volume total'][:,timepoint_to_plot]
contractile_volume = data_dict['contractile_volume'][:,timepoint_to_plot]
peak_kem_stand = data_dict['Peak KEM Stand'][:,timepoint_to_plot]
age = data_dict['Age'][:,timepoint_to_plot]
STS_time = data_dict['STS_time'][:,timepoint_to_plot]

# see if theres a nan in the row for either radial_diffusivity or peak_kem_stand. Remove that row from both vectors if theres a nan in that row of either vector
no_nans = ~(np.isnan(peak_kem_stand) | np.isnan(radial_diffusivity))
radial_diffusivity = radial_diffusivity[no_nans]
peak_kem_stand = -np.array(peak_kem_stand)[no_nans]
volume_total = volume_total[no_nans]
contractile_volume = contractile_volume[no_nans]
fat_fraction = fat_fraction[no_nans]
age = age[no_nans]
STS_time = STS_time[no_nans]

# multiple regression model to predict peak_kem_stand from some number of inputs
import statsmodels.api as sm
# create a design matrix
radial_diffusivity_standardized = (radial_diffusivity - np.mean(radial_diffusivity)) / np.std(radial_diffusivity)
volume_total_standardized = (volume_total - np.mean(volume_total)) / np.std(volume_total)
contractile_volume_standardized = (contractile_volume - np.mean(contractile_volume)) / np.std(contractile_volume)
X = np.column_stack((radial_diffusivity_standardized, volume_total_standardized))
X = sm.add_constant(X)
# fit the model
model = sm.OLS(peak_kem_stand, X)
results = model.fit()
# print the results
print(results.summary())


# compute correlation between peak_kem_stand and radial_diffusivity
x = radial_diffusivity_standardized + volume_total_standardized 
# x = radial_diffusivity_standardized + contractile_volume_standardized 

# x = results.fittedvalues
# standardize x to have mean 0 and std 1
# x = (x - np.mean(x)) / np.std(x)
# x = results.fittedvalues
# x = STS_time
correlation = np.corrcoef(peak_kem_stand, x)[0, 1]

# I want to statistically compare the strength of relationship between x and peak_kem_stand and x and STS_time
# I will use a Steiger’s z test to compare the correlation coefficients
from scipy.stats import norm
def steiger_z_test(r1, r2, n):
    # Calculate the z-score
    z = (r1 - r2) / np.sqrt((1 - r1**2) / (n - 2) + (1 - r2**2) / (n - 2))
    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(np.abs(z)))
    return z, p_value


# Make a subplot of 3 plots. All with KEM on the x axis. The y axes will be 
# 1. radial diffusivity
# 2. volume total
# 3. radial diffusivity * volume total

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Radial Diffusivity vs Peak KEM Stand
ax[0].scatter(peak_kem_stand, radial_diffusivity)
ax[0].set_xlabel('Peak KEM Stand')
ax[0].set_ylabel('Radial Diffusivity')
correlation1_kem, p_value1 = pearsonr(peak_kem_stand, radial_diffusivity)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1_kem, p_value1), transform=ax[0].transAxes, ha='center')

# Plot 2: Volume Total vs Peak KEM Stand
ax[1].scatter(peak_kem_stand, volume_total)
ax[1].set_xlabel('Peak KEM Stand')
ax[1].set_ylabel('Volume Total')
correlation2_kem, p_value2 = pearsonr(peak_kem_stand, volume_total)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2_kem, p_value2), transform=ax[1].transAxes, ha='center')

# Plot 3: Radial Diffusivity * Volume Total vs Peak KEM Stand
ax[2].scatter(peak_kem_stand, x)
ax[2].set_xlabel('Peak KEM Stand')
ax[2].set_ylabel('x (whatever you entered)')
correlation3_kem, p_value3 = pearsonr(peak_kem_stand, x)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3_kem, p_value3), transform=ax[2].transAxes, ha='center')

plt.show()

# Make a subplot of 3 plots. All with STS Time on the x axis. The y axes will be 
# 1. radial diffusivity
# 2. volume total
# 3. radial diffusivity * volume total

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Radial Diffusivity vs STS Time
ax[0].scatter(STS_time, radial_diffusivity)
ax[0].set_xlabel('STS Time')
ax[0].set_ylabel('Radial Diffusivity')
correlation1_sts, p_value1 = pearsonr(STS_time, radial_diffusivity)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1_sts, p_value1), transform=ax[0].transAxes, ha='center')

# Plot 2: Volume Total vs STS Time
ax[1].scatter(STS_time, volume_total)
ax[1].set_xlabel('STS Time')
ax[1].set_ylabel('Volume Total')
correlation2_sts, p_value2 = pearsonr(STS_time, volume_total)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2_sts, p_value2), transform=ax[1].transAxes, ha='center')

# Plot 3: Radial Diffusivity * Volume Total vs STS Time
ax[2].scatter(STS_time, x)
ax[2].set_xlabel('STS Time')
ax[2].set_ylabel('Radial Diffusivity * Volume Total')
correlation3_sts, p_value3 = pearsonr(STS_time, x)
ax[2].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation3_sts, p_value3), transform=ax[2].transAxes, ha='center')

plt.show()

# Make a 1x2 subplot with 1. kem vs x and 2. STS time vs x
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# Plot 1: Peak KEM Stand vs x
ax[0].scatter(peak_kem_stand, x)
ax[0].set_xlabel('Peak KEM Stand')
ax[0].set_ylabel('x (whatever you entered)')
correlation1, p_value1 = pearsonr(peak_kem_stand, x)
ax[0].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation1, p_value1), transform=ax[0].transAxes, ha='center')
# Plot 2: STS Time vs x
ax[1].scatter(STS_time, x)
ax[1].set_xlabel('STS Time')
ax[1].set_ylabel('x (whatever you entered)')
correlation2, p_value2 = pearsonr(STS_time, x)
ax[1].text(0.5, 0.9, 'r= {:.2f}, p= {:.2e}'.format(correlation2, p_value2), transform=ax[1].transAxes, ha='center')
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

# Calculate the z-score and p-value
z, p_value = steiger_z_test(correlation3_kem, correlation3_sts, len(x))
# Print the results
print(f"Steiger's z-test composite score: z = {z:.2f}, p-value = {p_value:.2e}")

# Calculate the steiger's test for radial diffusivity and volume total 
z, p_value = steiger_z_test(correlation1_kem, correlation1_sts, len(x))
# Print the results
#print(f"Steiger's z-test radial diffusivity: z = {z:.2f}, p-value = {p_value:.2e}")
# Calculate the steiger's test for volume total
z, p_value = steiger_z_test(correlation2_kem, correlation2_sts, len(x))
# Print the results
#print(f"Steiger's z-test volume total: z = {z:.2f}, p-value = {p_value:.2e}")

# I want predicted x values for kem and STS time, then compute the errors, and compare them using
# at t-test
from sklearn.linear_model import LinearRegression
# create a linear regression model
model = LinearRegression()
# fit the model to the data
model.fit(peak_kem_stand.reshape(-1, 1), x)
# get the predicted values
predicted_kem = model.predict(peak_kem_stand.reshape(-1, 1))
model2 = LinearRegression()
# fit the model to the data
model2.fit(STS_time.reshape(-1, 1), x)
# get the predicted values
predicted_sts = model2.predict(STS_time.reshape(-1, 1))
# compute the errors
error_kem = np.abs(x - predicted_kem)
error_sts = np.abs(x - predicted_sts)
print(error_kem)
print(error_sts)
# paired samples t-test
from scipy.stats import ttest_rel
# Perform the paired t-test
t_stat, p_value = ttest_rel(error_kem, error_sts)
# Print the results
print(f"t-statistic: {t_stat:.2f}, p-value: {p_value:.4e}")

