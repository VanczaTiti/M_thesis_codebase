import csv

import numpy as np
import os
import re
from matplotlib import pyplot as plt
from functions import spherical_to_cartesian, process_file, get_grd_files, get_grd_files_grid

r = 4.84
f = 24e9
c = 299792458  # m/s
# E_i = 0.000617156554001989  # incident field strength for r=4.84 m
E_i = -0.6051659119E-3 -0.1210637485E-3j  # incident field strength for r=4.84 m


def process_folder(base_path):
    grd_paths = get_grd_files(base_path)
    sig = []
    for grd in grd_paths:
        sig.append(process_file(grd, mask=[-1])[0])
        # sig.append(process_file(grd, mask=[-5])[0])
    return np.array(sig)

def process_folder_grid(base_path):
    grd_paths = get_grd_files_grid(base_path)
    sig = []
    for grd in grd_paths:
        # sig.append(process_file(grd, mask=[-9, -7, -3], polarisation=pol))  # use for 3*3 grid
        sig.append(process_file(grd, mask=[-5, -4, -2], polarisation=0))  # use for 3*3 grid
        # sig.append(process_file(grd, mask=[-4, -3, -2], polarisation=pol))  # use for 2*2 grid

    return np.array(sig)



def process_txt(path, line):
    """
    Extract numbers from a specified line in a text file.

    Parameters:
    path (str): The path to the text file.
    line (int): The line number to process (1-based index).

    Returns:
    np.ndarray: An array of numbers (real or complex) from the specified line.
    """
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            if line < 1 or line > len(lines):
                raise ValueError(f"Line number {line} is out of range. The file has {len(lines)} lines.")

            target_line = lines[line - 1].strip()

            if "i" in target_line:
                complex_numbers = [str2complex(num) for num in target_line.split("\t")]
                return np.array(complex_numbers, dtype=complex)
            else:
                real_numbers = [float(num) for num in target_line.split()]
                return np.array(real_numbers, dtype=float)

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path {path} was not found.")

    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the file: {e}")


def str2complex(s):
    nums = s.replace('- ', '-').replace('+ ', '-').replace('i', '').split()
    return complex(float(nums[0]), float(nums[1]))


def sin_cos_fit_residuals(data, phi):
    """
    Calculate the sinusoidal + constant fit of a 1D NumPy array and return the residuals.
    """
    if data.ndim != 1 or phi.ndim != 1:
        raise ValueError("Both 'data' and 'phi' must be 1D arrays.")
    if len(data) != len(phi):
        raise ValueError("'data' and 'phi' must have the same length.")

    # Convert phi from degrees to radians
    phi_rad = np.radians(phi)

    # Design matrix for the model a*sin(phi) + b*cos(phi) + c
    A = np.column_stack((np.sin(phi_rad), np.cos(phi_rad), np.ones_like(phi)))

    # Solve the least squares problem to find a, b, and c
    coefficients, _, _, _ = np.linalg.lstsq(A, data, rcond=None)

    # Calculate the fit
    fit = A @ coefficients

    # Calculate residuals (data - fit)
    residuals = data - fit

    return residuals

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Try_80_100_85_95"
# received = process_folder_grid(folder_path)
# # received = np.flip(received.reshape(21, 11, 3), 1)[:, :, 0]
# received = received.reshape(21, 11, 3)[:, :, 0]
# theta = np.linspace(-10, +10, 21, True)
# psi = np.linspace(+5, -5, 11, True)
# # RCS_SIM = 20 * np.log10(np.abs(received)/np.abs(0.1138130187E-02 - 0.9662416040E-03j) * 2 * 2 * np.sqrt(np.pi))
# RCS_SIM = 20 * np.log10(np.abs(received))+75
#
# plt.imshow(20 * np.log10(np.abs(received[:, :] / (0.1138130187E-02 - 0.9662416040E-03j) * r * 2 * np.sqrt(np.pi))),
#                     extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                     origin='upper', aspect='equal', cmap='viridis')
# plt.show()



# Example usage
sim_folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Measurement_recreation_try_4_hole\ps_1"
# sim_folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Measurement_recreation_try_3_p4deg\ps_1"
# sim_folder_path2 = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Measurement_recreation_try_3_p4deg\ps_1"
# sim_folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Measurement_recreation_try_2\ps_1"

RCS_meas_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\DTU_Measurement\Processed\RCS_ClubHead.txt"
S11_meas_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\DTU_Measurement\Processed\S11_ClubHead.txt"

# RCS_meas_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\DTU_Measurement\Processed\RCS_ClubHead_LongWindow.txt"
# S11_meas_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\DTU_Measurement\Processed\S11_ClubHead_LongWindow.txt"

line = 250
S11_meas = process_txt(S11_meas_path, line)
rcs_meas = 10 * np.log10(process_txt(RCS_meas_path, line))
psi_meas = np.linspace(+10, -10, 41, True) + 4.5
ang_meas = sin_cos_fit_residuals(np.unwrap(np.angle(S11_meas)), psi_meas)


received_sim = process_folder(sim_folder_path)
psi_sim = np.linspace(-10, +10, 41, True)
k = 2*np.pi/ (c / f)
A = 3*1E-7
B = -90
bg = A * np.exp(-1j * (np.sin(np.deg2rad(psi_sim))*0.05858 * 2 * k + np.deg2rad(B)))
received_sim2 = received_sim - bg
rcs_sim = 20 * np.log10(np.abs(received_sim/E_i * r * 2 * np.sqrt(np.pi)))
rcs_sim2 = 20 * np.log10(np.abs(received_sim2/E_i * r * 2 * np.sqrt(np.pi)))
rcs_bg = 20 * np.log10(np.abs(bg/E_i * r * 2 * np.sqrt(np.pi)))
ang_sim = sin_cos_fit_residuals(np.unwrap(np.angle(received_sim)), psi_sim)
ang_sim2 = sin_cos_fit_residuals(np.unwrap(np.angle(received_sim2)), psi_sim)


# received_sim2 = process_folder(sim_folder_path2)
# psi_sim = np.linspace(-10, +10, 41, True)
# rcs_sim2 = 20 * np.log10(np.abs(received_sim2/E_i * r * 2 * np.sqrt(np.pi)))
# ang_sim2 = np.unwrap(np.angle(received_sim))

fig, axs = plt.subplots(2, figsize=(6, 6))

# x
axs[0].set_title('RCS')
axs[0].plot(psi_sim, rcs_sim, 'g', label="sim")
axs[0].plot(psi_sim, rcs_sim2, 'b', label="sim2")
axs[0].plot(psi_sim, rcs_bg, 'b', label="bg")
# axs[0].plot(psi_sim, rcs_sim2, 'b', label="sim2")
# for i, th in enumerate(theta):
#     if np.abs(th) < 4:
#         axs[0].plot(psi, RCS_SIM[i, :], label=th)
axs[0].plot(psi_meas, rcs_meas, 'r', label="meas")
axs[0].set_xlabel('Psi [deg]')
axs[0].set_ylabel('RCS [dBsm]')
axs[0].legend()
axs[0].grid()

# y
axs[1].set_title('Angle')
axs[1].plot(psi_sim, ang_sim, 'g', label="sim")
axs[1].plot(psi_sim, ang_sim, 'b', label="sim2")
axs[1].plot(psi_meas, ang_meas, 'r', label="meas")
axs[1].set_xlabel('Psi [deg]')
axs[1].set_ylabel('angle [rad]')
axs[0].legend()
axs[1].grid()

plt.show()

data = np.column_stack((np.real(psi_sim),
                        np.real(rcs_sim),
                        np.flip(np.real(rcs_meas)),
                        np.real(ang_sim),
                        np.flip(np.real(ang_meas))))

# Column names
columns = ['rot_angles', 'rcs_sim',  'rcs_meas', 'ang_sim', 'ang_meas']

# Save to CSV
with open('Meas_sim.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)  # Write column headers
    writer.writerows(data)  # Write data rows
