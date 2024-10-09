import numpy as np
import os
import re

from matplotlib import pyplot as plt

d = 1.5/100
r = 2
f = 24e9

c = 299792458  # m/s
k = f*2*np.pi/c  # rad/m


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
folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\ps_1"


received = process_folder(folder_path)
print(received)

zero_phase = np.angle(np.exp(1j*r**2*k))

phase_center = np.array((-np.unwrap(np.angle(received[:, 0]) + zero_phase)/k/2,
                         d / 2 + r / d / k * np.unwrap(np.angle(received[:, 1]) - np.angle(received[:, 0])),
                         d / 2 + r / d / k * np.unwrap(np.angle(received[:, 2]) - np.angle(received[:, 0])))).T

y = np.linspace(0, 0.1, 20, True)

fig, axs = plt.subplots(4, figsize=(9, 9))

# x
axs[0].set_title('X Coordinate error')
axs[0].plot(y * 100, phase_center[:, 0] * 100, 'g')
axs[0].set_xlabel('dx [cm]')
axs[0].set_ylabel('Phase Center X [cm]')
axs[0].grid()

# y
axs[1].set_title('Y Coordinate error')
axs[1].plot(y * 100, (phase_center[:, 1]-y) * 100, 'g')
axs[1].set_xlabel('dx [cm]')
axs[1].set_ylabel('Phase Center Y [cm]')
axs[1].grid()

# y
axs[2].set_title('Y Coordinate error')
axs[2].plot(y * 100, phase_center[:, 2] * 100, 'g')
axs[2].set_xlabel('dx [cm]')
axs[2].set_ylabel('Phase Center Y [cm]')
axs[2].grid()


# power
p_max = np.max(received[:, 0])
axs[3].set_title('Normalised received power')
axs[3].plot(y * 100, 20 * np.log10(np.abs(received[:, 0]) / p_max), 'r')
axs[3].set_xlabel('dx [cm]')
axs[3].set_ylabel('Power [dB]')
axs[3].grid()

fig2, axs2 = plt.subplots(2, figsize=(9, 9))
axs2[0].plot(y * 100, np.angle(received[:, 0]), 'r')
axs2[0].plot(y * 100, np.angle(received[:, 1]), 'g')
axs2[0].plot(y * 100, np.angle(received[:, 2]), 'b')
axs2[0].grid()

axs2[1].plot(y * 100, np.angle(received[:, 1]) - np.angle(received[:, 0]), 'g')
axs2[1].plot(y * 100, np.angle(received[:, 2]) - np.angle(received[:, 0]), 'b')
axs2[1].grid()

plt.show()
pass
