import numpy as np
import matplotlib.pyplot as plt
import astropy.io.ascii as ascii
from scipy.optimize import curve_fit

#------------------------------- Load data -------------------------------

data_1 = ascii.read('Data Files/Nuclear_dataset_1')

t = np.array(data_1['Time'])
r1 = np.array(data_1['R1'])
e1 = np.array(data_1['e_R1'])
r2 = np.array(data_1['R2'])
e2 = np.array(data_1['e_R2'])
r3 = np.array(data_1['R3'])
e3 = np.array(data_1['e_R3'])
r4 = np.array(data_1['R4'])
e4 = np.array(data_1['e_R4'])

data_2 = ascii.read('Data Files/Nuclear_dataset_2')

t_2 = np.array(data_2['Time'])
r1_2 = np.array(data_2['R1'])
e1_2 = np.array(data_2['e_R1'])
r2_2 = np.array(data_2['R2'])
e2_2 = np.array(data_2['e_R2'])
r3_2 = np.array(data_2['R3'])
e3_2 = np.array(data_2['e_R3'])
r4_2 = np.array(data_2['R4'])
e4_2 = np.array(data_2['e_R4'])











# #============================First data set ==============================

# # Weighted least squares fit function 

def wls_fit(x, y, y_err):

    "This funciton uses the equations provided to determine the best fit gradient and intercept values for the data provided "

    mask = (y > 0) & (y_err > 0) # mask necessary as when bootstrap method is applied artificial data sets may contain negative valuues which logs cant be applied to
    x = x[mask]
    y = y[mask]
    y_err = y_err[mask]

    sigma_y = y_err / y # propagate errors for the natural log being taken of the y data
    y = np.log(y)

    S = np.sum(1 / sigma_y**2)
    S_x = np.sum(x / sigma_y**2)
    S_y = np.sum(y / sigma_y**2)
    S_xx = np.sum(x**2 / sigma_y**2)
    S_xy = np.sum(x * y / sigma_y**2)

    m_fit = (S * S_xy - S_x * S_y) / (S * S_xx - S_x**2)
    c_fit = (S_xx * S_y - S_x * S_xy) / (S * S_xx - S_x**2)


    return m_fit, sigma_y, c_fit

# #------------------------------- Compute WLS fits for each detector to get lambda and initial activity -------------------------------

lambda1, sigma_ln_A1, ln_c1 = wls_fit(t, r1, e1)
lambda2, sigma_ln_A2, ln_c2 = wls_fit(t, r2, e2)
lambda3, sigma_ln_A3, ln_c3 = wls_fit(t, r3, e3)
lambda4, sigma_ln_A4, ln_c4 = wls_fit(t, r4, e4)

# # -------------------------------Convert to positive lambda values -------------------------------

# # lambda1 = -lambda1 
# # lambda2 = -lambda2
# # lambda3 = -lambda3
# # lambda4 = -lambda4 

# #------------------------------- Calculate initial activities -------------------------------

A0_1 = np.exp(ln_c1) 
A0_2 = np.exp(ln_c2) 
A0_3 = np.exp(ln_c3) 
A0_4 = np.exp(ln_c4) 

#------------------------------- Plot WLS fit for detector 1 as example -------------------------------

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(t, A0_1 * np.exp(lambda1*t), '-', label='WLS Fit Detector 1', color='blue')
ax[0].scatter(t, r1, marker='x', label='Data Detector 1', color='black', s=20)
ax[0].errorbar(t, r1, yerr=e1, fmt='o', markersize=3, alpha=0.6)
ax[0].set_ylabel('Counts')
ax[0].set_title('Lambda value fit to Data for Detector 1 Data set 1')
ax[0].legend()

ax[1].plot(t, lambda1*t + ln_c1, '-', label='WLS Fit Detector 1', color='blue')
ax[1].scatter(t, np.log(r1), marker='x', label='Data Detector 1', color='black', s=20)
ax[1].errorbar(t, np.log(r1), yerr=sigma_ln_A1, fmt='o', markersize=3, alpha=0.6)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('ln(Counts)')
ax[1].set_title('Weighted Least Squares Fit for Detector 1 Data set 1')
ax[1].legend()
plt.show()



residuals_1 = r1 - A0_1 * np.exp(lambda1 * t)
residuals_2 = r2 - A0_2 * np.exp(-lambda2 * t)
residuals_3 = r3 - A0_3 * np.exp(-lambda3 * t)
residuals_4 = r4 - A0_4 * np.exp(-lambda4 * t)

# ------------------------------- plot residuals for detector 1 as an example  -------------------------------

plt.figure(figsize=(8,5))
plt.errorbar(t, residuals_1, yerr = e1, fmt = 'o')
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel("Time")
plt.ylabel("Residuals (counts)")
plt.title("Residuals for Detector 1")
plt.grid(True)
plt.show()

#------------------------------- Bootsrap function for lambda and initial activity uncertainties -------------------------------

def bootstrap_decay(t, err, lambda_fit, initial_activity_fit, residuals, Method, detector_num):

    "This function generates a range of artificial data sets for the single isotope model and applies the WLS function to find the lambda and initial activites of these artificial data sets."
    "This leaves you with a variety of initial activity and lambda values from which errors can be propagated via standard deviaiton or by finding the 16th and 84th percentiles"
    
    n = 50000
    lambda_samples = []
    activity_samples = []

    model = initial_activity_fit * np.exp(-lambda_fit * t)

    for iteration in range(n):

        idx = np.random.randint(0, len(residuals), len(residuals))
        r_boot = model + residuals[idx]
        err_boot = err[idx]

        lambda_boot, _, c_boot = wls_fit(t, r_boot, err_boot)
        lambda_boot = -lambda_boot
        a_boot = np.exp(c_boot)

        lambda_samples.append(lambda_boot)
        activity_samples.append(a_boot)

    # Method 1 - Standard Deviation

    stdev_lambda = np.std(lambda_samples)
    mean_lambda = np.mean(lambda_samples)
    stdev_activity = np.std(activity_samples)

    # Method 2 - Percentiles

    median_lambda = np.median(lambda_samples)
    percentile_16 = np.percentile(lambda_samples, 16)
    percentile_84 = np.percentile(lambda_samples, 84)

    plt.figure()
    plt.hist(lambda_samples, bins=80)
    plt.xlabel("λ samples")
    plt.ylabel("Count")
    plt.title(f"Bootstrap Distribution of λ for detetctor {detector_num}")

    if Method == 1:
        plt.axvline(mean_lambda, color='r', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(mean_lambda + stdev_lambda, color='g', linestyle='dashed', linewidth=1, label='Mean ± 1σ')
        plt.axvline(mean_lambda - stdev_lambda, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
    elif Method == 2:
        plt.axvline(median_lambda, color='r', linestyle='dashed', linewidth=1, label='Median')
        plt.axvline(percentile_16, color='g', linestyle='dashed', linewidth=1, label='16th & 84th Percentiles')
        plt.axvline(percentile_84, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
    
    plt.show()

    return stdev_lambda, stdev_activity

# #------------------------------- Get bootstrap uncertainties for each detector -------------------------------

stdev_lambda1, stdev_activity1 = bootstrap_decay(t, e1, lambda1, A0_1, residuals_1, 1, 1)
stdev_lambda2, stdev_activity2 = bootstrap_decay(t, e2, lambda2, A0_2, residuals_2, 1, 2)
stdev_lambda3, stdev_activity3 = bootstrap_decay(t, e3, lambda3, A0_3, residuals_3, 1, 3)
stdev_lambda4, stdev_activity4 = bootstrap_decay(t, e4, lambda4, A0_4, residuals_4, 1, 4)


#------------------------------- Weighted least squares fit function to get alpha value from power law for data set 1 -------------------------------

A0_values = np.array([A0_1, A0_2, A0_3, A0_4])
A0_errors = np.array([stdev_activity1, stdev_activity2, stdev_activity3, stdev_activity4])
distances = np.array([0.05, 0.1, 0.18, 0.3])
distances_err = np.array([0.0005, 0.0005, 0.0005, 0.0005])
ln_distances = np.log(distances)

alpha_fit, sigma_lnA0, lnC_fit = wls_fit(ln_distances, A0_values, A0_errors)
C = np.exp(lnC_fit)

#------------------------------- Bootsrap function for power law uncertainties -------------------------------

def bootstrap_powerlaw(distances, A0_values, A0_errors, alpha_fit, C_fit):

    "This function generates a range of artificial data sets for the power law model and applies the WLS function to find the best fit alpha value for these artificial data sets."
    "This leaves you with a variety of alpha from which errors are  propagated via standard deviaiton "

    n = 50000
    alpha_samples = []

    model = C_fit * (distances ** alpha_fit)
    
    residuals = A0_values - model
    
    for iteration in range(n):
        idx = np.random.randint(0, len(residuals), len(residuals))
        A0_boot = model + residuals[idx]
        A0_err_boot = A0_errors[idx] 
        
        alpha_boot, _, lnC_boot = wls_fit(ln_distances, A0_boot, A0_err_boot)
        alpha_samples.append(alpha_boot)

    # Method 1 - standard deviation

    stdev_alpha = np.std(alpha_samples) 

    # plot bootstrap distribution

    plt.figure()
    plt.hist(alpha_samples, bins=80)
    plt.xlabel("Alpha samples")
    plt.ylabel("Count")
    plt.title("Bootstrap Distribution of Alpha for data set 1")
    plt.axvline(alpha_fit, color='r', linestyle='dashed', linewidth=1, label='Alpha Fit')
    plt.axvline(alpha_fit + stdev_alpha, color='g', linestyle='dashed', linewidth=1, label='Alpha Fit ± 1σ')
    plt.axvline(alpha_fit - stdev_alpha, color='g', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()

    return stdev_alpha

#------------------------------- Get bootstrap uncertainty for alpha -------------------------------

stdev_alpha = bootstrap_powerlaw(distances, A0_values, A0_errors, alpha_fit, C)

print('\n','========================= Power Law Fit Results data set 1 =========================')
print('\n'+f'Alpha value data set 1 : {alpha_fit:.5g} ± {stdev_alpha:.5g}')

#------------------------------- Calculate N0 at t = t0 -------------------------------

detector_areas = np.array([0.0003142, 0.0003142, 0.0003142, 0.0003142]) 
detector_areas_err = np.array([0.001e-4, 0.001e-4, 0.001e-4, 0.001e-4])
distances = np.array([0.05, 0.1, 0.18, 0.3])
distances_err = np.array([0.0005, 0.0005, 0.0005, 0.0005])
A0_values = np.array([A0_1, A0_2, A0_3, A0_4])

detector_fluxes = A0_values / detector_areas

actual_A0_1 = detector_fluxes[0] * 4 * np.pi * (distances[0]**2)
actual_A0_2 = detector_fluxes[1] * 4 * np.pi * (distances[1]**2)
actual_A0_3 = detector_fluxes[2] * 4 * np.pi * (distances[2]**2)
actual_A0_4 = detector_fluxes[3] * 4 * np.pi * (distances[3]**2)

actual_N0_1 =  actual_A0_1 / lambda1
actual_N0_2 =  actual_A0_2 / lambda2
actual_N0_3 =  actual_A0_3 / lambda3
actual_N0_4 =  actual_A0_4 / lambda4

#------------------------------- propagate uncertainties for N0 -------------------------------

def propagate_N0_uncertainty(A0, A0_err, lambda_, lambda_err, d, d_err, area, area_err, N0):

    "This function uses the propagation of errors equation relating to fimnding the number of atoms present in the sample"
    "combines errors arising from initial activity calculations, decay constant calculations, uncertainty in detector area and uncertainty in detector distance from sample"

    term1 = (lambda_err / lambda_)**2
    term2 = (A0_err / A0)**2
    term3 = (2*d_err / d)**2
    term4 = (area_err / area)**2
    return N0 * np.sqrt(term1 + term2 + term3 + term4)

#------------------------------- Calculate N0 uncertainties for each detector -------------------------------

N0_1_uncertainty = propagate_N0_uncertainty(actual_A0_1, stdev_activity1, lambda1, stdev_lambda1, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_1)
N0_2_uncertainty = propagate_N0_uncertainty(actual_A0_2, stdev_activity2, lambda2, stdev_lambda2, distances[1], distances_err[1], detector_areas[1], detector_areas_err[1], actual_N0_2)
N0_3_uncertainty = propagate_N0_uncertainty(actual_A0_3, stdev_activity3, lambda3, stdev_lambda3, distances[2], distances_err[2], detector_areas[2], detector_areas_err[2], actual_N0_3)
N0_4_uncertainty = propagate_N0_uncertainty(actual_A0_4, stdev_activity4, lambda4, stdev_lambda4, distances[3], distances_err[3], detector_areas[3], detector_areas_err[3], actual_N0_4)

print('\n','========================= N0 data set 1 =========================')
print('\n'+f"Detector 1 N0: {actual_N0_1:.5g} ± {N0_1_uncertainty:.5g}")
print(f"Detector 2 N0: {actual_N0_2:.5g} ± {N0_2_uncertainty:.5g}")
print(f"Detector 3 N0: {actual_N0_3:.5g} ± {N0_3_uncertainty:.5g}")
print(f"Detector 4 N0: {actual_N0_4:.5g} ± {N0_4_uncertainty:.5g}")

#------------------------------- plot N0 values ------------------------------- 

detectors = [1, 2, 3, 4]
N0_values = [-actual_N0_1, -actual_N0_2, -actual_N0_3, -actual_N0_4]
N0_errors = [-N0_1_uncertainty, -N0_2_uncertainty, -N0_3_uncertainty, -N0_4_uncertainty]

plt.errorbar(detectors, N0_values, yerr=N0_errors, fmt='o', capsize=5, markersize=6, color='royalblue', ecolor='royalblue')
plt.xlabel("Detector Number")
plt.ylabel("N₀ (atoms)")
plt.title("N0 per Detector")
plt.grid(True)
plt.xticks(detectors)
plt.show()

#------------------------------- Curve Fit for single isotope model to compare with WLS results -------------------------------

def single_isotope(t, A0, lam):
    return A0 * np.exp(-lam * t)

p0_detector1_values = [A0_1, lambda1]
p0_detector2_values = [A0_2, lambda2]
p0_detector3_values = [A0_3, lambda3]
p0_detector4_values = [A0_4, lambda4]

popt1, pcov1 = curve_fit(single_isotope, t, r1, sigma=e1, p0=p0_detector1_values, absolute_sigma=False)
curve_fit_A0_1, curve_fit_lambda_1 = popt1

popt2, pcov2 = curve_fit(single_isotope, t, r2, sigma=e2, p0=p0_detector2_values, absolute_sigma=False)
curve_fit_A0_2, curve_fit_lambda_2 = popt2

popt3, pcov3 = curve_fit(single_isotope, t, r3, sigma=e3, p0=p0_detector3_values, absolute_sigma=False)
curve_fit_A0_3, curve_fit_lambda_3 = popt3

popt4, pcov4 = curve_fit(single_isotope, t, r4, sigma=e4, p0=p0_detector4_values, absolute_sigma=False)
curve_fit_A0_4, curve_fit_lambda_4 = popt4

print('\n','========================= Curve Fit Results vs WLS Results data set 1 =========================')
print('\n'+f"From curve_fit detector 1, A0 = {curve_fit_A0_1:.2f}, λ = {curve_fit_lambda_1:.4f} vs WLS: A0 = {A0_1:.2f} ± {stdev_activity1:.2f}, λ = {lambda1:.4f} ± {stdev_lambda1:.4f}")
print(f"From curve_fit detector 2, A0 = {curve_fit_A0_2:.2f}, λ = {curve_fit_lambda_2:.4f} vs WLS: A0 = {A0_2:.2f} ± {stdev_activity2:.2f}, λ = {lambda2:.4f} ± {stdev_lambda2:.4f}")
print(f"From curve_fit detector 3, A0 = {curve_fit_A0_3:.2f}, λ = {curve_fit_lambda_3:.4f} vs WLS: A0 = {A0_3:.2f} ± {stdev_activity3:.2f}, λ = {lambda3:.4f} ± {stdev_lambda3:.4f}")
print(f"From curve_fit detector 4, A0 = {curve_fit_A0_4:.2f}, λ = {curve_fit_lambda_4:.4f} vs WLS: A0 = {A0_4:.2f} ± {stdev_activity4:.2f}, λ = {lambda4:.4f} ± {stdev_lambda4:.4f}")

#------------------------------- Plot comparison of WLS and curve_fit results -------------------------------

detectors = np.array([1, 2, 3, 4])
lambda_wls = np.array([-lambda1, -lambda2, -lambda3, -lambda4])
lambda_wls_err = np.array([stdev_lambda1, stdev_lambda2, stdev_lambda3, stdev_lambda4])
lambda_cf = np.array([curve_fit_lambda_1, curve_fit_lambda_2, curve_fit_lambda_3, curve_fit_lambda_4])
A0_wls = np.array([A0_1, A0_2, A0_3, A0_4])
A0_wls_err = np.array([stdev_activity1, stdev_activity2, stdev_activity3, stdev_activity4])
A0_cf = np.array([curve_fit_A0_1, curve_fit_A0_2, curve_fit_A0_3, curve_fit_A0_4])  

fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

ax[0].errorbar(detectors, lambda_wls, yerr=lambda_wls_err, fmt='o', capsize=4, label='WLS fit')
ax[0].scatter(detectors, lambda_cf, marker='x', color='black', label='curve_fit')
ax[0].set_ylabel('λ (s^-1)')
ax[0].set_title('WLS fit vs curve_fit for lambda and A0 values')
ax[0].legend()

ax[1].errorbar(detectors, A0_wls, yerr=A0_wls_err, fmt='o', capsize=4, label='WLS fit')
ax[1].scatter(detectors, A0_cf, marker='x', color='black', label='curve_fit')
ax[1].set_xlabel('Detector')
ax[1].set_ylabel('A0 (Bq)')
ax[1].set_xticks(detectors)
ax[1].legend()
plt.tight_layout()
plt.show()

# ------------------------------ weighted average of decay constant ----------------------------

lambda_values = np.array([lambda1, lambda2, lambda3, lambda4])
lambda_uncertainties = np.array([stdev_lambda1, stdev_lambda2, stdev_lambda3, stdev_lambda4])

w_i = 1/(lambda_uncertainties**2)
weighted_av_decay_constant = (np.sum(lambda_values * w_i))/(np.sum(w_i))
weighted_av_decay_constant_err = np.sqrt(1/(np.sum(w_i)))

print('\n'+'==================== Average decay constant data set 1 ===================')
print('\n'+f"λ = {weighted_av_decay_constant:.4f} ± {weighted_av_decay_constant_err:.4f}")

# ------------------------------ half life calculations ------------------------------

av_half_life_isotope = np.log(2)/weighted_av_decay_constant
av_half_life_isotope_err = av_half_life_isotope * np.sqrt((weighted_av_decay_constant_err/weighted_av_decay_constant)**2)

half_life_det1 = np.log(2)/lambda1
half_life_err_det1 = half_life_det1 * (stdev_lambda1/lambda1)

half_life_det2 = np.log(2)/lambda2
half_life_err_det2 = half_life_det2 * (stdev_lambda2/lambda2)

half_life_det3 = np.log(2)/lambda3
half_life_err_det3 = half_life_det3 * (stdev_lambda3/lambda3)

half_life_det4 = np.log(2)/lambda4
half_life_err_det4 = half_life_det4 * (stdev_lambda4/lambda4)

print('\n'+"==================== Half life isotope dataset 1 ====================")
print('\n'+f" average half life = ({av_half_life_isotope:.2f} ± {av_half_life_isotope_err:.2f}) seconds") 
print(f"Detector 1 half life: {half_life_det1:.5g} ± {half_life_err_det1:.5g}")
print(f"Detector 2 half life: {half_life_det2:.5g} ± {half_life_err_det2:.5g}")
print(f"Detector 3 half life: {half_life_det3:.5g} ± {half_life_err_det3:.5g}")
print(f"Detector 4 half life: {half_life_det4:.5g} ± {half_life_err_det4:.5g}")


# 19.2 ± 1.5 seconds meaning either 139 Eu or 166 W

# -------------------------- Plot half lifes of detectors to compare with appendix values ----------------------------

half_lives = [-half_life_det1,-half_life_det2,-half_life_det3,-half_life_det4,-av_half_life_isotope]
half_life_errors = [half_life_err_det1,half_life_err_det2,half_life_err_det3,half_life_err_det4,-av_half_life_isotope_err]
labels = ["Det1","Det2","Det3","Det4","Weighted Avg"]

x = np.arange(len(labels))

plt.figure(figsize=(8, 5))
plt.errorbar(x, half_lives, yerr=half_life_errors, fmt='o', capsize = 5)
plt.xticks(x, labels)
plt.ylabel("Half-life (s)")
plt.title("Detector Half-Lives and Weighted Average with Error Bars")
plt.axhline(y=17.9, color='orangered', linestyle='--', label="139 Eu")
plt.axhline(y=19.2, color='royalblue', linestyle='--', label="166 W")
plt.legend()
plt.tight_layout()
plt.show()







#============================ Second data set ============================

#------------------------------- Compute WLS fits for each detector to get lambda and initial activity -------------------------------

lambda1_2, sigma_ln_A1_2, ln_c1_2 = wls_fit(t_2, r1_2, e1_2)
lambda2_2, sigma_ln_A2_2, ln_c2_2 = wls_fit(t_2, r2_2, e2_2)
lambda3_2, sigma_ln_A3_2, ln_c3_2 = wls_fit(t_2, r3_2, e3_2)
lambda4_2, sigma_ln_A4_2, ln_c4_2 = wls_fit(t_2, r4_2, e4_2)

lambda1_2 = -lambda1_2
lambda2_2 = -lambda2_2
lambda3_2 = -lambda3_2
lambda4_2 = -lambda4_2 

A0_1_2 = np.exp(ln_c1_2) 
A0_2_2 = np.exp(ln_c2_2) 
A0_3_2 = np.exp(ln_c3_2) 
A0_4_2 = np.exp(ln_c4_2) 

#------------------------------- Get bootstrap uncertainties for each detector -------------------------------

residuals_1_2 = r1_2 - A0_1_2 * np.exp(-lambda1_2 * t_2)
residuals_2_2 = r2_2 - A0_2_2 * np.exp(-lambda2_2 * t_2)
residuals_3_2 = r3_2 - A0_3_2 * np.exp(-lambda3_2 * t_2)
residuals_4_2 = r4_2 - A0_4_2 * np.exp(-lambda4_2 * t_2)

# -------------------------------------- plot residual graph as example --------------------------------------

plt.figure(figsize=(8,5))
plt.errorbar(t, residuals_1_2, yerr = e1, fmt = 'o')
plt.axhline(0, color='black', linestyle='--', linewidth=1)

plt.xlabel("Time")
plt.ylabel("Residuals (counts)")
plt.title("Residuals for Detector 1")
plt.legend()
plt.grid(True)
plt.show()

stdev_lambda1_2, stdev_activity1_2 = bootstrap_decay(t_2, e1_2, lambda1_2, A0_1_2, residuals_1_2, 1, 1)
stdev_lambda2_2, stdev_activity2_2 = bootstrap_decay(t_2, e2_2, lambda2_2, A0_2_2, residuals_2_2, 1, 2)
stdev_lambda3_2, stdev_activity3_2 = bootstrap_decay(t_2, e3_2, lambda3_2, A0_3_2, residuals_3_2, 1, 3)
stdev_lambda4_2, stdev_activity4_2 = bootstrap_decay(t_2, e4_2, lambda4_2, A0_4_2, residuals_4_2, 1, 4)

print('\n','========================= WLS results lambda and inital activity data set 2 =========================')
print('\n'+f"Detector 1 data set 2 Decay Constant: ({lambda1_2:.4f}) ± {stdev_lambda1_2:.4f}) and initial activity: ({A0_1_2:.2f} ± {stdev_activity1_2:.2f})")
print(f"Detector 2 data set 2 Decay Constant: ({lambda2_2:.4f} ± {stdev_lambda2_2:.4f}) and initial activity: ({A0_2_2:.2f} ± {stdev_activity2_2:.2f})")
print(f"Detector 3 data set 2 Decay Constant: ({lambda3_2:.4f} ± {stdev_lambda3_2:.4f}) and initial activity: ({A0_3_2:.2f} ± {stdev_activity3_2:.2f})")
print(f"Detector 4 data set 2 Decay Constant: ({lambda4_2:.4f} ± {stdev_lambda4_2:.4f}) and initial activity: ({A0_4_2:.2f} ± {stdev_activity4_2:.2f})")


#------------------------------- Curve Fit for single isotope model to compare with WLS results -------------------------------

def single_isotope(t, A0, lam):
    return A0 * np.exp(-lam * t)

p0_detector1_values_2 = [A0_1_2, lambda1_2]
p0_detector2_values_2 = [A0_2_2, lambda2_2]
p0_detector3_values_2 = [A0_3_2, lambda3_2]
p0_detector4_values_2 = [A0_4_2, lambda4_2]

popt1_2, pcov1_2 = curve_fit(single_isotope, t_2, r1_2, sigma=e1_2, p0=p0_detector1_values_2, absolute_sigma=False)
curve_fit_A0_1_2, curve_fit_lambda_1_2 = popt1_2

popt2_2, pcov2_2 = curve_fit(single_isotope, t_2, r2_2, sigma=e2_2, p0=p0_detector2_values_2, absolute_sigma=False)
curve_fit_A0_2_2, curve_fit_lambda_2_2 = popt2_2

popt3_2, pcov3_2 = curve_fit(single_isotope, t_2, r3_2, sigma=e3_2, p0=p0_detector3_values_2, absolute_sigma=False)
curve_fit_A0_3_2, curve_fit_lambda_3_2 = popt3_2

popt4_2, pcov3_2 = curve_fit(single_isotope, t_2, r4_2, sigma=e4_2, p0=p0_detector4_values_2, absolute_sigma=False)
curve_fit_A0_4_2, curve_fit_lambda_4_2 = popt4_2

print('\n','========================= Curve Fit Results vs WLS Results data set 2 =========================')
print('\n'+f"From curve_fit detector 1, A0 = {curve_fit_A0_1_2:.2f}, λ = {curve_fit_lambda_1_2:.4f} vs WLS: A0 = {A0_1_2:.2f} ± {stdev_activity1_2:.2f}, λ = {lambda1_2:.4f} ± {stdev_lambda1_2:.4f}")
print(f"From curve_fit detector 2, A0 = {curve_fit_A0_2_2:.2f}, λ = {curve_fit_lambda_2_2:.4f} vs WLS: A0 = {A0_2_2:.2f} ± {stdev_activity2_2:.2f}, λ = {lambda2_2:.4f} ± {stdev_lambda2_2:.4f}")
print(f"From curve_fit detector 3, A0 = {curve_fit_A0_3_2:.2f}, λ = {curve_fit_lambda_3_2:.4f} vs WLS: A0 = {A0_3_2:.2f} ± {stdev_activity3_2:.2f}, λ = {lambda3_2:.4f} ± {stdev_lambda3_2:.4f}")
print(f"From curve_fit detector 4, A0 = {curve_fit_A0_4_2:.2f}, λ = {curve_fit_lambda_4_2:.4f} vs WLS: A0 = {A0_4_2:.2f} ± {stdev_activity4_2:.2f}, λ = {lambda4_2:.4f} ± {stdev_lambda4_2:.4f}")

#------------------------------- Plot WLS fit for detector 1 as example -------------------------------

fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(t_2, A0_1_2 * np.exp(lambda1_2*t_2), '-', label='WLS Fit Detector 1', color='blue')
ax[0].scatter(t_2, r1_2, marker='x', label='Data Detector 1', color='black', s=20)
ax[0].errorbar(t_2, r1_2, yerr=e1_2, fmt='o', markersize=3, alpha=0.6)
ax[0].set_ylabel('Counts')
ax[0].set_title('Lambda value fit to Data for Detector 1 Data set 2')
ax[0].legend()

ax[1].plot(t_2, lambda1_2*t_2 + ln_c1_2, '-', label='WLS Fit Detector 1', color='blue')
ax[1].scatter(t_2, np.log(r1_2), marker='x', label='Data Detector 1', color='black', s=20)
ax[1].errorbar(t_2, np.log(r1_2), yerr=sigma_ln_A1_2, fmt='o', markersize=3, alpha=0.6)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('ln(Counts)')
ax[1].set_title('Weighted Least Squares Fit for Detector 1 Data set 2')
ax[1].legend()
plt.show()

#------------------------------- Plot comparison of WLS and curve_fit results -------------------------------

detectors = np.array([1, 2, 3, 4])
lambda_wls_2 = np.array([-lambda1_2, -lambda2_2, -lambda3_2, -lambda4_2])
lambda_wls_err_2 = np.array([stdev_lambda1_2, stdev_lambda2_2, stdev_lambda3_2, stdev_lambda4_2])
lambda_cf_2 = np.array([curve_fit_lambda_1_2, curve_fit_lambda_2_2, curve_fit_lambda_3_2, curve_fit_lambda_4_2])
A0_wls_2 = np.array([A0_1_2, A0_2_2, A0_3_2, A0_4_2])
A0_wls_err_2 = np.array([stdev_activity1_2, stdev_activity2_2, stdev_activity3_2, stdev_activity4_2])
A0_cf_2 = np.array([curve_fit_A0_1_2, curve_fit_A0_2_2, curve_fit_A0_3_2, curve_fit_A0_4_2])

fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

ax[0].errorbar(detectors, lambda_wls_2, yerr=lambda_wls_err_2, fmt='o', capsize=4, label='WLS fit')
ax[0].scatter(detectors, lambda_cf_2, marker='x', color='black', label='curve_fit')
ax[0].set_ylabel('λ (s^-1)')
ax[0].set_title('WLS fit vs curve_fit for lambda and A0 values Data set 2')
ax[0].legend()

ax[1].errorbar(detectors, A0_wls_2, yerr=A0_wls_err_2, fmt='o', capsize=4, label='WLS fit')
ax[1].scatter(detectors, A0_cf_2, marker='x', color='black', label='curve_fit')
ax[1].set_xlabel('Detector')
ax[1].set_ylabel('A0 (Bq)')
ax[1].set_xticks(detectors)
ax[1].legend()

plt.tight_layout()
plt.show()

#------------------------------- Weighted least squares fit function to get alpha value from power law for data set 2 -------------------------------

A0_values_2 = np.array([A0_1_2, A0_2_2, A0_3_2, A0_4_2])
A0_errors_2 = np.array([stdev_activity1_2, stdev_activity2_2, stdev_activity3_2, stdev_activity4_2])
distances = np.array([0.05, 0.1, 0.18, 0.3])
distances_err = np.array([0.0005, 0.0005, 0.0005, 0.0005])
ln_distances = np.log(distances)

alpha_fit_2, sigma_lnA0_2, lnC_fit_2 = wls_fit(ln_distances, A0_values_2, A0_errors_2)
C_2 = np.exp(lnC_fit_2)

#------------------------------- Get bootstrap uncertainty for alpha -------------------------------

stdev_alpha_2 = bootstrap_powerlaw(distances, A0_values_2, A0_errors_2, alpha_fit_2, C_2)

print('\n','========================= Power Law Fit Results data set 2 =========================')
print('\n'+f'Alpha value data set 2 : {alpha_fit_2:.5g} ± {stdev_alpha_2:.5g}')

#------------------------------- Plot both power law linear fits -------------------------------

# dataset 1
plt.scatter(ln_distances, np.log(A0_values_2), marker='x', label='Initial Activities Data set 2', color='orange', s=20)
plt.errorbar(ln_distances, np.log(A0_values_2), yerr=A0_errors_2/A0_values_2, fmt='o', markersize=3, alpha=0.6, ecolor='orange', markeredgecolor='orange', markerfacecolor='orange')
plt.plot(ln_distances, alpha_fit_2*ln_distances + lnC_fit_2, '-', label='Power Law Fit Data set 2', color='orange')

# dataset 2
plt.scatter(ln_distances, np.log(A0_values), marker='x', label='Initial Activities Data set 1', color='royalblue', s=20)
plt.errorbar(ln_distances, np.log(A0_values), yerr=A0_errors/A0_values, fmt='o', markersize=3, alpha=0.6, ecolor='royalblue', markeredgecolor='royalblue', markerfacecolor='royalblue')
plt.plot(ln_distances, alpha_fit*ln_distances + lnC_fit, '-', label='Power Law Fit Data set 1', color='royalblue')

plt.xlabel('Distance (m)')
plt.ylabel('ln(Initial Activity A0 (Bq))')
plt.title('Power Law Fit Comparison between Data set 1 and Data set 2')
plt.legend()
plt.show()


#------------------------------- Chi-squared grid calculation and heat map plotting -------------------------------

def chi_squared_grid(t_2, r_2, e_2, lambda_fit, A0_fit, lambda_range, A0_range, detector_num):

    "This function defines a range of lambda and initial activity values for the single isotope model centered on lambda_fit and A0_fit, it then calculates a chi^2 value corresponding to each combination"
    "The minimum chi^2 value is identified along with the corresponding lambda and initial activity values"
    "A heat map is also plot with confidence intervals for two parameters as errors in order to map this chi^2 fit"

    # chi-squared grid calculation

    lambda_values = np.linspace(lambda_fit - lambda_range, lambda_fit + lambda_range, 100)
    A0_values = np.linspace(A0_fit - A0_range, A0_fit + A0_range, 100)

    chi_squared = np.zeros((len(lambda_values), len(A0_values)))

    for i, lam in enumerate(lambda_values):
        for j, A0 in enumerate(A0_values):
            model = A0 * np.exp(-lam * t_2)
            chi_squared[i, j] = np.sum(((r_2 - model) / e_2) ** 2)

    min_chi_2_index = np.unravel_index(np.argmin(chi_squared), chi_squared.shape)
    lambda_best_fit = lambda_values[min_chi_2_index[0]]
    A0_best_fit = A0_values[min_chi_2_index[1]]
    min_chi_2 = np.min(chi_squared)

    # propagate uncertainties
    # two free parameters
    # delta chi-squared = 2.30 for 1-sigma, 6.17 for 2-sigma, 11.8 for 3-sigma

    sigma_level_1 = min_chi_2 + 2.30
    sigma_level_2 = min_chi_2 + 6.17
    sigma_level_3 = min_chi_2 + 11.8

    # Plot heat map

    plt.figure()
    plt.pcolormesh(A0_values, lambda_values, chi_squared, cmap='gist_heat')
    plt.plot(A0_best_fit, lambda_best_fit, 'x', color='cyan', markersize=10, label='Best Fit')
    plt.colorbar(label='Chi-squared')
    plt.xlabel('Initial Activity A0')
    plt.ylabel('Decay Constant λ')
    plt.title(f'Chi-squared heat map for Detector {detector_num} Data Set 2')
    plt.legend()

    # Contour lines for 1σ, 2σ, 3σ
    contours1 = plt.contour(A0_values, lambda_values, chi_squared, levels=[sigma_level_1], colors='white')
    contours2 = plt.contour(A0_values, lambda_values, chi_squared, levels=[sigma_level_2], colors='white')
    contours3 = plt.contour(A0_values, lambda_values, chi_squared, levels=[sigma_level_3], colors='white')

    plt.clabel(contours1, fmt='1σ', inline=True, fontsize=10)
    plt.clabel(contours2, fmt='2σ', inline=True, fontsize=10)
    plt.clabel(contours3, fmt='3σ', inline=True, fontsize=10)

    # value errors from chi-squared grid
    # mask for 1-sigma contour
    mask_1sigma = chi_squared <= sigma_level_1

    # get lambda and A0 values within 1-sigma region
    lambda_inside = lambda_values[np.any(mask_1sigma, axis=1)]
    A0_inside = A0_values[np.any(mask_1sigma, axis=0)]

    # calculate errors
    lambda_err_plus = np.max(lambda_inside) 
    lambda_err_minus = np.min(lambda_inside)
    A0_err_plus = np.max(A0_inside) 
    A0_err_minus = np.min(A0_inside)


    plt.show()

    return min_chi_2, lambda_best_fit, A0_best_fit, lambda_err_plus, lambda_err_minus, A0_err_plus, A0_err_minus

min_chi_2_det1, lambda_chi_2_1, A0_chi_2_1, lambda_chi_2_err_plus_1, lambda_chi_2_err_minus_1, A0_chi_2_err_plus_1, A0_chi_2_err_minus_1 = chi_squared_grid(t_2, r1_2, e1_2, lambda1_2, A0_1_2, 0.00075, 70, 1)
min_chi_2_det2, lambda_chi_2_2, A0_chi_2_2, lambda_chi_2_err_plus_2, lambda_chi_2_err_minus_2, A0_chi_2_err_plus_2, A0_chi_2_err_minus_2 = chi_squared_grid(t_2, r2_2, e2_2, lambda2_2, A0_2_2, 0.0015, 40, 2)
min_chi_2_det3, lambda_chi_2_3, A0_chi_2_3, lambda_chi_2_err_plus_3, lambda_chi_2_err_minus_3, A0_chi_2_err_plus_3, A0_chi_2_err_minus_3 = chi_squared_grid(t_2, r3_2, e3_2, lambda3_2, A0_3_2, 0.002, 20, 3)
min_chi_2_det4, lambda_chi_2_4, A0_chi_2_4, lambda_chi_2_err_plus_4, lambda_chi_2_err_minus_4, A0_chi_2_err_plus_4, A0_chi_2_err_minus_4  = chi_squared_grid(t_2, r4_2, e4_2, lambda4_2, A0_4_2, 0.005, 15, 4)

print('\n','========================= Chi-squared Grid single isotope Results data set 2 =========================')
print('\n'+f"Detector 1 data set 2 Chi-squared min: {min_chi_2_det1:.5g}, λ best fit: {lambda_chi_2_1:.5g} + {(lambda_chi_2_err_plus_1 - lambda_chi_2_1):.5g}/- {(lambda_chi_2_1 - lambda_chi_2_err_minus_1):.5g}, A0 best fit: {A0_chi_2_1:.5g} + {(A0_chi_2_err_plus_1 - A0_chi_2_1):.5g}/- {(A0_chi_2_1 - A0_chi_2_err_minus_1):.5g}")
print(f"Detector 2 data set 2 Chi-squared min: {min_chi_2_det2:.5g}, λ best fit: {lambda_chi_2_2:.5g} + {(lambda_chi_2_err_plus_2 - lambda_chi_2_2):.5g}/- {(lambda_chi_2_1 - lambda_chi_2_err_minus_2):.5g}, A0 best fit: {A0_chi_2_2:.5g} + {(A0_chi_2_err_plus_2 - A0_chi_2_2):.5g}/- {(A0_chi_2_2 - A0_chi_2_err_minus_2):.5g}")
print(f"Detector 3 data set 2 Chi-squared min: {min_chi_2_det3:.5g}, λ best fit: {lambda_chi_2_3:.5g} + {(lambda_chi_2_err_plus_3 - lambda_chi_2_3):.5g}/- {(lambda_chi_2_1 - lambda_chi_2_err_minus_3):.5g}, A0 best fit: {A0_chi_2_3:.5g} + {(A0_chi_2_err_plus_3 - A0_chi_2_3):.5g}/- {(A0_chi_2_3 - A0_chi_2_err_minus_3):.5g}")
print(f"Detector 4 data set 2 Chi-squared min: {min_chi_2_det4:.5g}, λ best fit: {lambda_chi_2_4:.5g} + {(lambda_chi_2_err_plus_4 - lambda_chi_2_4):.5g}/- {(lambda_chi_2_1 - lambda_chi_2_err_minus_4):.5g}, A0 best fit: {A0_chi_2_4:.5g} + {(A0_chi_2_err_plus_4 - A0_chi_2_4):.5g}/- {(A0_chi_2_4 - A0_chi_2_err_minus_4):.5g}")

# ------------------------------- AIC and BIC calculation for single isotope model -------------------------------

def AIC_and_BIC(min_chi_2, num_of_params, num_of_data_points):
    AIC = min_chi_2 + (2 * num_of_params)
    BIC = min_chi_2 + (num_of_params * np.log(num_of_data_points))
    return AIC, BIC

single_isotope_AIC_det1, single_isotope_BIC_det1 = AIC_and_BIC(min_chi_2_det1, 2, len(t_2))
single_isotope_AIC_det2, single_isotope_BIC_det2 = AIC_and_BIC(min_chi_2_det2, 2, len(t_2))
single_isotope_AIC_det3, single_isotope_BIC_det3 = AIC_and_BIC(min_chi_2_det3, 2, len(t_2))
single_isotope_AIC_det4, single_isotope_BIC_det4 = AIC_and_BIC(min_chi_2_det4, 2, len(t_2))

print('\n','========================= AIC and BIC single isotope Results data set 2 =========================')
print('\n'+f"Single isotope model detector 1 AIC: {single_isotope_AIC_det1:.5g}, BIC: {single_isotope_BIC_det1:.5g}")
print(f"Single isotope model detector 2 AIC: {single_isotope_AIC_det2:.5g}, BIC: {single_isotope_BIC_det2:.5g}")
print(f"Single isotope model detector 3 AIC: {single_isotope_AIC_det3:.5g}, BIC: {single_isotope_BIC_det3:.5g}")
print(f"Single isotope model detector 4 AIC: {single_isotope_AIC_det4:.5g}, BIC: {single_isotope_BIC_det4:.5g}")


# ------------------------------- Curve Fit for two isotope model -------------------------------

def two_isotope_2(t, A01_2, lambda1_2, A02_2, lambda2_2):
    return A01_2 * np.exp(-lambda1_2 * t) + A02_2 * np.exp(-lambda2_2 * t)

p0_detector1_2 = [1090, 0.04, 2770, 0.014]
p0_detector2_2 = [430, 0.03, 530, 0.012]
p0_detector3_2 = [63, 0.09, 255, 0.016]
p0_detector4_2 = [25, 0.12, 90, 0.0016]

popt1_2, pcov1_2 = curve_fit(two_isotope_2, t_2, r1_2, sigma=e1_2, p0=p0_detector1_2, absolute_sigma=False)
A0a_1_2, lambda1a_2, A0b_1_2, lambda1b_2 = popt1_2

popt2_2, pcov2_2 = curve_fit(two_isotope_2, t_2, r2_2, sigma=e2_2, p0=p0_detector2_2, absolute_sigma=False)
A0a_2_2, lambda2a_2, A0b_2_2, lambda2b_2 = popt2_2

popt3_2, pcov3_2 = curve_fit(two_isotope_2, t_2, r3_2, sigma=e3_2, p0=p0_detector3_2, absolute_sigma=False)
A0a_3_2, lambda3a_2, A0b_3_2, lambda3b_2 = popt3_2

popt4_2, pcov4_2 = curve_fit(two_isotope_2, t_2, r4_2, sigma=e4_2,p0=p0_detector4_2, absolute_sigma=False)
A0a_4_2, lambda4a_2, A0b_4_2, lambda4b_2 = popt4_2



print('\n','========================= Curve Fit Results two isotope model data set 2 =========================')
print('\n'+f"From curve_fit detector 1 data set 2, A01 = {A0a_1_2:.5g}, λ1a = {lambda1a_2:.5g}, A02 = {A0b_1_2:.5g}, λ1b = {lambda1b_2:.5g}")
print(f"From curve_fit detector 2 data set 2, A01 = {A0a_2_2:.5g}, λ2a = {lambda2a_2:.5g}, A02 = {A0b_2_2:.5g}, λ2b = {lambda2b_2:.5g}")
print(f"From curve_fit detector 3 data set 2, A01 = {A0a_3_2:.5g}, λ3a = {lambda3a_2:.5g}, A02 = {A0b_3_2:.5g}, λ3b = {lambda3b_2:.5g} ")
print(f"From curve_fit detector 4 data set 2, A01 = {A0a_4_2:.5g}, λ4a = {lambda4a_2:.5g}, A02 = {A0b_4_2:.5g}, λ4b = {lambda4b_2:.5g} ")  


# # ------------------------------- 4D Chi-squared grid calculation for two isotope model -------------------------------

def chi_squared_grid_two_isotope(t_2, r_2, e_2, lambdaa_fit, A0a_fit, lambdab_fit, A0b_fit, lambdaa_range, A0a_range, lambdab_range, A0b_range, detector_num):

    "This function defines a range of lambda and initial activity values for the double isotope model centered on lambdaa_fit, lambdab_fit and A0a_fit and A0b_fit. it then calculates a chi^2 value corresponding to each combination"
    "The minimum chi^2 value is identified along with the corresponding lambda and initial activity values"
    "errors ariser from confidence intervals with four parameters"

    # chi-squared grid calculation

    lambda1a_values = np.linspace(lambdaa_fit - lambdaa_range, lambdaa_fit + lambdaa_range, 25)
    A01_values = np.linspace(A0a_fit - A0a_range, A0a_fit + A0a_range, 25)
    lambda2b_values = np.linspace(lambdab_fit - lambdab_range, lambdab_fit + lambdab_range, 25)
    A02_values = np.linspace(A0b_fit - A0b_range, A0b_fit + A0b_range, 25)

    chi_squared = np.zeros((len(lambda1a_values), len(A01_values), len(lambda2b_values), len(A02_values)))

    for i, lam1a in enumerate(lambda1a_values):
        for j, A01 in enumerate(A01_values):
            for k, lam2b in enumerate(lambda2b_values):
                for l, A02 in enumerate(A02_values):
                    model = A01 * np.exp(-lam1a * t_2) + A02 * np.exp(-lam2b * t_2)
                    chi_squared[i, j, k, l] = np.sum(((r_2 - model) / e_2) ** 2)

    min_chi_2_index = np.unravel_index(np.argmin(chi_squared), chi_squared.shape)
    lambdaa_best_fit = lambda1a_values[min_chi_2_index[0]]
    A0a_best_fit = A01_values[min_chi_2_index[1]]
    lambdab_best_fit = lambda2b_values[min_chi_2_index[2]]
    A0b_best_fit = A02_values[min_chi_2_index[3]]
    min_chi_2 = np.min(chi_squared)

    # propagate uncertainties
    # two free parameters
    # delta chi-squared = 4.72 for 1-sigma, 9.70 for 2-sigma, 16.3 for 3-sigma

    sigma_level_1 = min_chi_2 + 4.72
    sigma_level_2 = min_chi_2 + 9.70
    sigma_level_3 = min_chi_2 + 16.3

    # value errors from chi-squared grid
    # mask for 1-sigma contour
    mask_1sigma = chi_squared <= sigma_level_1

    # get lambda and A0 values within 1-sigma contour
    lambdaa_inside = lambda1a_values[np.any(mask_1sigma, axis=(1,2,3))]
    A0a_inside     = A01_values[np.any(mask_1sigma, axis=(0,2,3))]
    lambdab_inside = lambda2b_values[np.any(mask_1sigma, axis=(0,1,3))]
    A0b_inside     = A02_values[np.any(mask_1sigma, axis=(0,1,2))]

    # calculate errors
    lambdaa_err_plus = np.max(lambdaa_inside) - lambdaa_best_fit
    lambdaa_err_minus = lambdaa_best_fit -np.min(lambdaa_inside)
    A0a_err_plus = np.max(A0a_inside) - A0a_best_fit
    A0a_err_minus = A0a_best_fit - np.min(A0a_inside)
    lambdab_err_plus = np.max(lambdab_inside) - lambdab_best_fit
    lambdab_err_minus = lambdab_best_fit - np.min(lambdab_inside)
    A0b_err_plus = np.max(A0b_inside) - A0b_best_fit
    A0b_err_minus = A0b_best_fit - np.min(A0b_inside)

    return min_chi_2, lambdaa_best_fit, A0a_best_fit, lambdab_best_fit, A0b_best_fit, lambdaa_err_plus, lambdaa_err_minus, A0a_err_plus, A0a_err_minus, lambdab_err_plus, lambdab_err_minus, A0b_err_plus, A0b_err_minus

min_chi_2_det1_two_iso, lambda1a_best_fit, A01a_best_fit, lambda1b_best_fit, A01b_best_fit, lambda1a_err_plus, lambda1a_err_minus, A01a_err_plus, A01a_err_minus, lambda1b_err_plus, lambda1b_err_minus, A01b_err_plus, A01b_err_minus  = chi_squared_grid_two_isotope(t_2, r1_2, e1_2, lambda1a_2, A0a_1_2, lambda1b_2, A0b_1_2, 0.01, 290, 0.001, 300, 1)
min_chi_2_det2_two_iso, lambda2a_best_fit, A02a_best_fit, lambda2b_best_fit, A02b_best_fit, lambda2a_err_plus, lambda2a_err_minus, A02a_err_plus, A02a_err_minus, lambda2b_err_plus, lambda2b_err_minus, A02b_err_plus, A02b_err_minus  = chi_squared_grid_two_isotope(t_2, r2_2, e2_2, lambda2a_2, A0a_2_2, lambda2b_2, A0b_2_2, 0.02, 250, 0.01, 260, 2)
min_chi_2_det3_two_iso, lambda3a_best_fit, A03a_best_fit, lambda3b_best_fit, A03b_best_fit, lambda3a_err_plus, lambda3a_err_minus, A03a_err_plus, A03a_err_minus, lambda3b_err_plus, lambda3b_err_minus, A03b_err_plus, A03b_err_minus  = chi_squared_grid_two_isotope(t_2, r3_2, e3_2, lambda3a_2, A0a_3_2, lambda3b_2, A0b_3_2, 0.05, 15, 0.01, 15, 3)
min_chi_2_det4_two_iso, lambda4a_best_fit, A04a_best_fit, lambda4b_best_fit, A04b_best_fit, lambda4a_err_plus, lambda4a_err_minus, A04a_err_plus, A04a_err_minus, lambda4b_err_plus, lambda4b_err_minus, A04b_err_plus, A04b_err_minus  = chi_squared_grid_two_isotope(t_2, r4_2, e4_2, lambda4a_2, A0a_4_2, lambda4b_2, A0b_4_2, 0.08, 10, 0.01, 7, 4)

print('\n','========================= Chi-squared Grid two isotope Results data set 2 =========================')
print('\n'+f"Detector 1 dataset 2 min chi^2: {min_chi_2_det1_two_iso:.5g}, λ1a: {lambda1a_best_fit:.4f} + {lambda1a_err_plus:.4f}/- {lambda1a_err_minus:.4f}, A01a: {A01a_best_fit:.2f} + {A01a_err_plus:.2f}/- {A01a_err_minus:.2f}, λ1b: {lambda1b_best_fit:.4f} + {lambda1b_err_plus:.4f}/- {lambda1b_err_minus:.4f}, A01b: {A01b_best_fit:.2f} + {A01b_err_plus:.2f}/- {A01b_err_minus:.2f}")
print(f"Detector 2 dataset 2 min chi^2: {min_chi_2_det2_two_iso:.5g}, λ2a: {lambda2a_best_fit:.4f} + {lambda2a_err_plus:.4f}/- {lambda2a_err_minus:.4f}, A02a: {A02a_best_fit:.2f} + {A02a_err_plus:.2f}/- {A02a_err_minus:.2f}, λ2b: {lambda2b_best_fit:.4f} + {lambda2b_err_plus:.4f}/- {lambda2b_err_minus:.4f}, A02b: {A02b_best_fit:.2f} + {A02b_err_plus:.2f}/- {A02b_err_minus:.2f}")
print(f"Detector 3 dataset 2 min chi^2: {min_chi_2_det3_two_iso:.5g}, λ3a: {lambda3a_best_fit:.4f} + {lambda3a_err_plus:.4f}/- {lambda3a_err_minus:.4f}, A03a: {A03a_best_fit:.2f} + {A03a_err_plus:.2f}/- {A03a_err_minus:.2f}, λ3b: {lambda3b_best_fit:.4f} + {lambda3b_err_plus:.4f}/- {lambda3b_err_minus:.4f}, A03b: {A03b_best_fit:.2f} + {A03b_err_plus:.2f}/- {A03b_err_minus:.2f}")
print(f"Detector 4 dataset 2 min chi^2: {min_chi_2_det4_two_iso:.5g}, λ4a: {lambda4a_best_fit:.4f} + {lambda4a_err_plus:.4f}/- {lambda4a_err_minus:.4f}, A04a: {A04a_best_fit:.2f} + {A04a_err_plus:.2f}/- {A04a_err_minus:.2f}, λ4b: {lambda4b_best_fit:.4f} + {lambda4b_err_plus:.4f}/- {lambda4b_err_minus:.4f}, A04b: {A04b_best_fit:.2f} + {A04b_err_plus:.2f}/- {A04b_err_minus:.2f}")

# ------------------------------- AIC and BIC calculation for two isotope model -------------------------------

two_isotope_AIC_det1, two_isotope_BIC_det1 = AIC_and_BIC(min_chi_2_det1_two_iso, 4, len(t_2))
two_isotope_AIC_det2, two_isotope_BIC_det2 = AIC_and_BIC(min_chi_2_det2_two_iso, 4, len(t_2))   
two_isotope_AIC_det3, two_isotope_BIC_det3 = AIC_and_BIC(min_chi_2_det3_two_iso, 4, len(t_2))
two_isotope_AIC_det4, two_isotope_BIC_det4 = AIC_and_BIC(min_chi_2_det4_two_iso, 4, len(t_2))

print('\n'+f"Two isotope model detector 1 AIC: {two_isotope_AIC_det1:.5g}, BIC: {two_isotope_BIC_det1:.5g}")
print(f"Two isotope model detector 2 AIC: {two_isotope_AIC_det2:.5g}, BIC: {two_isotope_BIC_det2:.5g}") 
print(f"Two isotope model detector 3 AIC: {two_isotope_AIC_det3:.5g}, BIC: {two_isotope_BIC_det3:.5g}")
print(f"Two isotope model detector 4 AIC: {two_isotope_AIC_det4:.5g}, BIC: {two_isotope_BIC_det4:.5g}")

# double isotope model provides a better fit with AIC and BIC so assume its decay constants are more accurate

# ------------------------------ half life calculations ------------------------------

# find average decay constant uncertainties for each detector 

lambda1a_err = (lambda1a_err_plus + lambda1a_err_minus)/2
lambda2a_err = (lambda2a_err_plus + lambda2a_err_minus)/2
lambda3a_err = (lambda3a_err_plus + lambda3a_err_minus)/2
lambda4a_err = (lambda4a_err_plus + lambda4a_err_minus)/2

lambda1b_err = (lambda1b_err_plus + lambda1b_err_minus)/2
lambda2b_err = (lambda2b_err_plus + lambda2b_err_minus)/2
lambda3b_err = (lambda3b_err_plus + lambda3b_err_minus)/2
lambda4b_err = (lambda4b_err_plus + lambda4b_err_minus)/2

lambdaa_values = np.array([lambda1a_best_fit, lambda2a_best_fit, lambda3a_best_fit, lambda4a_best_fit])
lambdaa_uncertainies = np.array([lambda1a_err, lambda2a_err, lambda3a_err, lambda4a_err])

lambdab_values = np.array([lambda1b_best_fit, lambda2b_best_fit, lambda3b_best_fit, lambda4b_best_fit])
lambdab_uncertainties = np.array([lambda1b_err, lambda2b_err, lambda3b_err, lambda4b_err])

wa_i = 1/(lambdaa_uncertainies**2)
weighted_av_decay_constant_a = (np.sum(lambdaa_values*wa_i))/(np.sum(wa_i))
weighted_av_decay_constant_err_a = np.sqrt(1/(np.sum(wa_i)))

wb_i = 1/(lambdab_uncertainties**2)
weighted_av_decay_constant_b = (np.sum(lambdab_values*wb_i))/(np.sum(wb_i))
weighted_av_decay_constant_err_b = np.sqrt(1/(np.sum(wb_i)))

print('\n'+'==================== Average decay constants data set 2 ===================')
print('\n'+f"λa = {weighted_av_decay_constant_a:.4f} ± {weighted_av_decay_constant_err_a:.4f}")
print(f"λb = {weighted_av_decay_constant_b:.4f} ± {weighted_av_decay_constant_err_b:.4f}")

# average half life 

av_half_life_isotope_a = np.log(2)/weighted_av_decay_constant_a
av_half_life_isotope_err_a = av_half_life_isotope_a * np.sqrt((weighted_av_decay_constant_err_a/weighted_av_decay_constant_a)**2)

av_half_life_isotope_b = np.log(2)/weighted_av_decay_constant_b
av_half_life_isotope_err_b = av_half_life_isotope_b * np.sqrt((weighted_av_decay_constant_err_b/weighted_av_decay_constant_b)**2)

# half life for each detector 

half_life_det1a = np.log(2) / lambda1a_best_fit
lambda_err_det1a = (lambda1a_err_minus+lambda1a_err_plus)/2
half_life_err_det1a = half_life_det1a * np.sqrt((lambda_err_det1a / lambda1a_best_fit)**2)

half_life_det1b = np.log(2) / lambda1b_best_fit
lambda_err_det1b = (lambda1b_err_minus+lambda1b_err_plus)/2
half_life_err_det1b = half_life_det1b * np.sqrt((lambda_err_det1b / lambda1b_best_fit)**2)

half_life_det2a = np.log(2) / lambda2a_best_fit
lambda_err_det2a = (lambda2a_err_minus+lambda2a_err_plus)/2
half_life_err_det2a = half_life_det2a * np.sqrt((lambda_err_det2a / lambda2a_best_fit)**2)

half_life_det2b = np.log(2) / lambda2b_best_fit
lambda_err_det2b = (lambda2b_err_minus+lambda2b_err_plus)/2
half_life_err_det2b = half_life_det2b * np.sqrt((lambda_err_det2b / lambda2b_best_fit)**2)

half_life_det3a = np.log(2) / lambda3a_best_fit
lambda_err_det3a = (lambda3a_err_minus+lambda3a_err_plus)/2
half_life_err_det3a = half_life_det3a * np.sqrt((lambda_err_det3a / lambda3a_best_fit)**2)

half_life_det3b = np.log(2) / lambda3b_best_fit
lambda_err_det3b = (lambda3b_err_minus+lambda3b_err_plus)/2
half_life_err_det3b = half_life_det3b * np.sqrt((lambda_err_det3b / lambda3b_best_fit)**2)

half_life_det4a = np.log(2) / lambda4a_best_fit
lambda_err_det4a = (lambda4a_err_minus+lambda4a_err_plus)/2
half_life_err_det4a = half_life_det4a * np.sqrt((lambda_err_det4a / lambda4a_best_fit)**2)

half_life_det4b = np.log(2) / lambda4b_best_fit
lambda_err_det4b = (lambda4b_err_minus+lambda4b_err_plus)/2
half_life_err_det4b = half_life_det4b * np.sqrt((lambda_err_det4b / lambda4b_best_fit)**2)

print('\n'+"==================== Half life isotopes dataset 2 ====================")
print('\n'+f"average half life of isotope a = ({av_half_life_isotope_a:.2f} ± {av_half_life_isotope_err_a:.2f}) seconds") 
print(f"average half life of isotope b = ({av_half_life_isotope_b:.2f} ± {av_half_life_isotope_err_b:.2f}) seconds") 

print('\n'+f"half life det 1a: {half_life_det1a:.5g} ± {half_life_err_det1a:.5g}")
print(f"half life det 1b: {half_life_det1b:.5g} ± {half_life_err_det1b:.5g}")
print(f"half life det 2a: {half_life_det2a:.5g} ± {half_life_err_det2a:.5g}")
print(f"half life det 2b: {half_life_det2b:.5g} ± {half_life_err_det2b:.5g}")
print(f"half life det 3a: {half_life_det3a:.5g} ± {half_life_err_det3a:.5g}")
print(f"half life det 3b: {half_life_det3b:.5g} ± {half_life_err_det3b:.5g}")
print(f"half life det 4a: {half_life_det4a:.5g} ± {half_life_err_det4a:.5g}")
print(f"half life det 4b: {half_life_det4b:.5g} ± {half_life_err_det4b:.5g}")

# ------------------------------- plot graph of half lives compared to half lives of isotopes in appendix ----------------------------

# half life values for detectors
half_lives = [ half_life_det1a, half_life_det1b, half_life_det2a, half_life_det2b, half_life_det3a, half_life_det3b, half_life_det4a, half_life_det4b, av_half_life_isotope_a, av_half_life_isotope_b]

# errors
half_life_errors = [half_life_err_det1a, half_life_err_det1b, half_life_err_det2a, half_life_err_det2b, half_life_err_det3a, half_life_err_det3b, half_life_err_det4a, half_life_err_det4b, av_half_life_isotope_err_a, av_half_life_isotope_err_b]

# labels
labels = ["Det 1A", "Det 1B","Det 2A", "Det 2B","Det 3A", "Det 3B","Det 4A", "Det 4B","Weighted average A","Weighted average B"]

x = np.arange(len(labels))

plt.figure(figsize=(12, 6))
plt.errorbar(x, half_lives, yerr=half_life_errors, fmt='o', capsize = 5)
plt.xticks(x, labels, )
plt.ylabel("Half-life (s)")
plt.title("Detector Half-Lives and Weighted Averages with Error Bars")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.axhline(y=51, color='royalblue', linestyle='--', label="168 W")
plt.axhline(y=61.2, color='orangered', linestyle='--', label="120 Cs")
plt.axhline(y=17.9, color='orange', linestyle='--', label="139 Eu")
plt.axhline(y=19.2, color='green', linestyle='--', label="166 W")
plt.axhline(y=30.9, color='yellow', linestyle='--', label="124 Cs")
plt.axhline(y=8.4, color='purple', linestyle='--', label="137 Eu")
plt.axhline(y=12.1, color='pink', linestyle='--', label="138 Eu")
plt.legend()
plt.show()


#------------------------------- Calculate N0 at t = t0 -------------------------------

# get N0 for lambda a and lambda b and add together to get total atoms

detector_areas = np.array([0.0003142, 0.0003142, 0.0003142, 0.0003142]) 
detector_areas_err = np.array([0.001e-4, 0.001e-4, 0.001e-4, 0.001e-4])
distances = np.array([0.05, 0.1, 0.18, 0.3])
distances_err = np.array([0.0005, 0.0005, 0.0005, 0.0005])
A0a_values_2 = np.array([A01a_best_fit, A02a_best_fit, A03a_best_fit, A04a_best_fit])
A0b_values_2 = np.array([A01b_best_fit, A02b_best_fit, A03b_best_fit, A04b_best_fit,])

detector_fluxesa_2 = A0a_values_2 / detector_areas
detector_fluxesb_2 = A0b_values_2 / detector_areas

actual_A0a_1_2 = detector_fluxesa_2[0] * 4 * np.pi * (distances[0]**2)
actual_A0a_2_2 = detector_fluxesa_2[1] * 4 * np.pi * (distances[1]**2)
actual_A0a_3_2 = detector_fluxesa_2[2] * 4 * np.pi * (distances[2]**2)
actual_A0a_4_2 = detector_fluxesa_2[3] * 4 * np.pi * (distances[3]**2)

actual_A0b_1_2 = detector_fluxesb_2[0] * 4 * np.pi * (distances[0]**2)
actual_A0b_2_2 = detector_fluxesb_2[1] * 4 * np.pi * (distances[1]**2)
actual_A0b_3_2 = detector_fluxesb_2[2] * 4 * np.pi * (distances[2]**2)
actual_A0b_4_2 = detector_fluxesb_2[3] * 4 * np.pi * (distances[3]**2)

actual_N0_1a_2 =  actual_A0a_1_2 / lambda1a_best_fit
actual_N0_2a_2 =  actual_A0a_2_2 / lambda2a_best_fit
actual_N0_3a_2 =  actual_A0a_3_2 / lambda3a_best_fit
actual_N0_4a_2 =  actual_A0a_4_2 / lambda4a_best_fit

actual_N0_1b_2 =  actual_A0b_1_2 / lambda1b_best_fit
actual_N0_2b_2 =  actual_A0b_2_2 / lambda2b_best_fit
actual_N0_3b_2 =  actual_A0b_3_2 / lambda3b_best_fit
actual_N0_4b_2 =  actual_A0b_4_2 / lambda4b_best_fit

A01a_err = (A01a_err_plus + A01a_err_minus) / 2
A02a_err = (A02a_err_plus + A02a_err_minus) / 2
A03a_err = (A03a_err_plus + A03a_err_minus) / 2
A04a_err = (A04a_err_plus + A04a_err_minus) / 2

A01b_err = (A01b_err_plus + A01b_err_minus) / 2
A02b_err = (A02b_err_plus + A02b_err_minus) / 2
A03b_err = (A03b_err_plus + A03b_err_minus) / 2
A04b_err = (A04b_err_plus + A04b_err_minus) / 2

N0_1a_uncertainty_2 = propagate_N0_uncertainty(actual_A0a_1_2, A01a_err, lambda1a_best_fit, lambda1a_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_1a_2)
N0_2a_uncertainty_2 = propagate_N0_uncertainty(actual_A0a_2_2, A02a_err, lambda2a_best_fit, lambda2a_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_2a_2)
N0_3a_uncertainty_2 = propagate_N0_uncertainty(actual_A0a_3_2, A03a_err, lambda3a_best_fit, lambda3a_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_3a_2)
N0_4a_uncertainty_2 = propagate_N0_uncertainty(actual_A0a_4_2, A04a_err, lambda4a_best_fit, lambda4a_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_4a_2)

N0_1b_uncertainty_2 = propagate_N0_uncertainty(actual_A0b_1_2, A01b_err, lambda1b_best_fit, lambda1b_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_1b_2)
N0_2b_uncertainty_2 = propagate_N0_uncertainty(actual_A0b_2_2, A02b_err, lambda2b_best_fit, lambda2b_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_2b_2)
N0_3b_uncertainty_2 = propagate_N0_uncertainty(actual_A0b_3_2, A03b_err, lambda3b_best_fit, lambda3b_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_3b_2)
N0_4b_uncertainty_2 = propagate_N0_uncertainty(actual_A0b_4_2, A04b_err, lambda4b_best_fit, lambda4b_err, distances[0], distances_err[0], detector_areas[0], detector_areas_err[0], actual_N0_4b_2)

total_N0_1_2 = actual_N0_1a_2 + actual_N0_1b_2
N0_1_err_2 = N0_1a_uncertainty_2 + N0_1b_uncertainty_2

total_N0_2_2 = actual_N0_2a_2 + actual_N0_2b_2
N0_2_err_2 = N0_2a_uncertainty_2 + N0_2b_uncertainty_2

total_N0_3_2 = actual_N0_3a_2 + actual_N0_3b_2
N0_3_err_2 = N0_3a_uncertainty_2 + N0_3b_uncertainty_2

total_N0_4_2 = actual_N0_4a_2 + actual_N0_4b_2
N0_4_err_2 = N0_4a_uncertainty_2 + N0_4b_uncertainty_2

print('\n'+"==================== N0 values dataset 2 ====================")
print('\n'+f"Detector 1 N0 = {total_N0_1_2:.5g} ± {N0_1_err_2:.5g}")
print(f"Detector 2 N0 = {total_N0_2_2:.5g} ± {N0_2_err_2:.5g}")
print(f"Detector 3 N0 = {total_N0_3_2:.5g} ± {N0_3_err_2:.5g}")
print(f"Detector 4 N0 = {total_N0_4_2:.5g} ± {N0_4_err_2:.5g}")

#------------------------------- plot N0 values ------------------------------- 

detectors = [1, 2, 3, 4]
N0_values = [total_N0_1_2, total_N0_2_2, total_N0_3_2, total_N0_4_2]
N0_errors = [N0_1_err_2, N0_2_err_2, N0_3_err_2, N0_4_err_2]

plt.errorbar(detectors, N0_values, yerr=N0_errors, fmt='o', capsize=5, markersize=6, color='royalblue', ecolor='royalblue')
plt.xlabel("Detector Number")
plt.ylabel("N₀ (atoms)")
plt.title("N0 per Detector")
plt.grid(True)
plt.xticks(detectors)
plt.show()


# ---------------------------- plot single vs double model values and residuals for detector 1 ---------------------------

# double values 

lambda_a = 0.0426
A0_a = 1093.27
lambda_b = 0.0145
A0_b = 2771.57

# single values

lambda_single = 0.0174
A0_single = 3637.00

# Double model
A_double = A0_a * np.exp(-lambda_a * t) + A0_b * np.exp(-lambda_b * t)

# single model
A_single = A0_single * np.exp(-lambda_single * t)

# residuals 
double_residuals = r1_2 - A_double
single_residuals = r1_2 - A_single

# Plot 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)
ax1.errorbar(t, r1_2, yerr=e1_2, fmt='x', color='black', label='Data', capsize=3)
ax1.plot(t, A_double, label='Double Exponential Fit', color='blue')
ax1.plot(t, A_single, label='Single Exponential Fit', color='orange')
ax1.set_ylabel('A (counts)')
ax1.set_title('Double isotope model vs single isotope model')
ax1.grid(True)
ax1.legend()

ax2.errorbar(t, double_residuals, yerr=e1_2, fmt='o', color='blue', label='Double Residuals', capsize=3)
ax2.errorbar(t, single_residuals, yerr=e1_2, fmt='o', color='orange', label='Single Residuals', capsize=3)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)  
ax2.set_xlabel('Time')
ax2.set_ylabel('Residuals (counts)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Please note this code was altered and tweaked during the process of making the slides for the presentation so when values are printed the signs may not be as you would assume, i.e may print negative half lifes etc but values are correct 
