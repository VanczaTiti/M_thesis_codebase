import csv

import numpy as np
import os
import re
from matplotlib import pyplot as plt
from functions import spherical_to_cartesian, process_file, get_grd_files

d = 1.5/100
r = 20
f = 24e9
c = 299792458  # m/s
k = f*2*np.pi/c  # rad/m
# E_i = -8.429668E1  # incident field strenth dB
E_i = 0.1138130187E-02 - 0.9662416040E-03j  # incident field r=2

text = ""


def process_folder(base_path):
    grd_paths = get_grd_files(base_path)
    sig = []
    for grd in grd_paths:
        sig.append(process_file(grd))

    return np.array(sig)


# Example usage
# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\corner_ref_" + text + "05_65_mm"
folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Job_03"


received = process_folder(folder_path)
zero_phase = np.angle(np.exp(1j*r**2*k))

phase_center_sph = np.array((-np.unwrap(np.angle(received[:, 0]) + zero_phase)/k/2 + r,
                             np.arcsin(np.unwrap(np.angle(received[:, 1]) - np.angle(received[:, 0])) / k / d),
                             np.arcsin(np.unwrap(np.angle(received[:, 2]) - np.angle(received[:, 0])) / k / d))).T

phase_center = spherical_to_cartesian(phase_center_sph) + np.array((0, d/2, d/2))

# y = np.linspace(0, 0, 20, True)
a = np.linspace(0.5, 10, len(phase_center), True)
# a_rel = np.linspace(0.005, 0.065, len(phase_center), True) * f / c

fig, axs = plt.subplots(4, figsize=(9, 9))

# x
axs[0].set_title('X Coordinate error')
axs[0].plot(a, (phase_center[:, 0] - r) * 100, 'g')
# axs[0].plot(y / 0.0125, np.angle(received[:, 0]), 'g')

axs[0].set_xlabel('dx [cm]')
axs[0].set_ylabel('Phase Center X [cm]')
axs[0].grid()

# y
axs[1].set_title('Y Coordinate error')
axs[1].plot(a, (phase_center[:, 1]) * 100, 'g')
axs[1].set_xlabel('side length [cm]')
axs[1].set_ylabel('[cm]')
axs[1].grid()

# y
axs[2].set_title('Y Coordinate error')
axs[2].plot(a, phase_center[:, 2] * 100, 'g')
axs[2].set_xlabel('side length [cm]')
axs[2].set_ylabel('[cm]')
axs[2].grid()


# power
# signal_max = np.max(received[:, 0])
axs[3].set_title('RCS [dB (m2)]')
axs[3].plot(a, 20 * np.log10(np.abs(received[:, 0]/E_i*np.sqrt(r**2*4*np.pi))), 'r')
axs[3].plot(a, 10 * np.log10((a/100)**4*3.14/3/0.0125**2), 'g')
axs[3].set_xlabel('side length [cm]')
# axs[3].set_ylabel('Power [dB]')
axs[3].set_ylabel('Power [dBm]')
axs[3].grid()

# fig2, axs2 = plt.subplots(2, figsize=(9, 9))
# axs2[0].plot(y * 100, np.angle(received[:, 0]), 'r')
# axs2[0].plot(y * 100, np.angle(received[:, 1]), 'g')
# axs2[0].plot(y * 100, np.angle(received[:, 2]), 'b')
# axs2[0].grid()
#
# axs2[1].plot(y * 100, np.angle(received[:, 1]) - np.angle(received[:, 0]), 'g')
# axs2[1].plot(y * 100, np.angle(received[:, 2]) - np.angle(received[:, 0]), 'b')
# axs2[1].grid()

plt.show()

data = np.column_stack((a,
                        (phase_center[:, 0]-r) * 100,
                        phase_center[:, 1] * 100,
                        phase_center[:, 2] * 100,
                        # np.abs(20 * np.log10(np.abs(received[:, 0]) / signal_max))))
                        20 * np.log10(np.abs(received[:, 0]/E_i*np.sqrt(r**2*4*np.pi)))))


# Column names
columns = ['size', text + 'ph_c_err_x', text + 'ph_c_err_y', text + 'ph_c_err_z', text + 'RCS']

# Save to CSV
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)  # Write column headers
    writer.writerows(data)    # Write data rows

print("Done")
