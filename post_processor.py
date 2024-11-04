import csv

import numpy as np
import os
import re
from matplotlib import pyplot as plt

d = 1.5/100
r = 20
f = 24e9
c = 299792458  # m/s
k = f*2*np.pi/c  # rad/m
# E_i = -8.429668E1  # incident field strenth dB
E_i = 0.1164005044E-03 - 0.9359531396E-04j  # incident field r=20
# E_i = 0.1138130187E-02 - 0.9662416040E-03j  # incident field r=2

text = ""


def spherical_to_cartesian(spherical_coords):
    r = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]  # azimuthal angle (-180 to +180)
    phi = spherical_coords[:, 2]    # polar angle (-90 to +90)
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta) + d / 2
    z = r * np.sin(phi) + d / 2
    cartesian_coords = np.vstack((x, y, z)).T
    return cartesian_coords


def process_file(file_path):
    complex_numbers = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Loop through the last 4 lines that contain the data
        for line in lines[-4:-1]:
            # Split the line into individual numbers
            numbers = line.split()
            # Extract the first and fourth numbers, combine into a complex number
            real_part = float(numbers[0])
            imag_part = float(numbers[1])
            complex_num = complex(real_part, imag_part)
            complex_numbers.append(complex_num)
    return complex_numbers


def get_number_from_folder(folder_name):
    # This regex looks for a pattern like "_number" at the end of the folder name
    match = re.search(r'_(\d+)$', folder_name)
    return int(match.group(1)) if match else None


def get_grd_files(base_path):
    grd_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".grd"):
                full_path = os.path.join(root, file)
                # Extract the folder number from the folder containing the .grd file
                folder_name = os.path.basename(os.path.dirname(full_path))
                folder_number = get_number_from_folder(folder_name)
                if folder_number is not None:
                    grd_files.append((full_path, folder_number))

    # Sort the list by the folder number
    grd_files.sort(key=lambda x: x[1])

    # Return only the file paths, not the folder numbers
    return [file_path for file_path, _ in grd_files]


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

phase_center = spherical_to_cartesian(phase_center_sph)

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
