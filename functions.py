import os
import numpy as np
import re
from scipy.signal import resample


def spherical_to_cartesian(spherical_coords):
    if spherical_coords.ndim == 2:
        r = spherical_coords[:, 0]
        theta = spherical_coords[:, 1]  # azimuth angle (-180 to +180)
        psi = spherical_coords[:, 2]    # polar angle (-90 to +90)
    elif spherical_coords.ndim == 3:
        r = spherical_coords[:, :, 0]
        theta = spherical_coords[:, :, 1]  # azimuth angle (-180 to +180)
        psi = spherical_coords[:, :, 2]  # polar angle (-90 to +90)
    else:
        print('dim error')
    x = r * np.cos(psi) * np.cos(theta)
    y = r * np.cos(psi) * np.sin(theta)
    z = r * np.sin(psi)
    if spherical_coords.ndim == 2:
        cartesian_coords = np.vstack((x, y, z)).T
    elif spherical_coords.ndim == 3:
        cartesian_coords = np.stack((x, y, z), axis=-1)
    return cartesian_coords


def process_file(file_path, mask=None):
    if mask is None:
        mask = [-4, -3, -2]
    complex_numbers = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Loop through the last 4 lines that contain the data
        for line in [lines[m] for m in mask]:
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


def get_grd_files_grid(base_path):
    grd_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".grd"):
                full_path = os.path.join(root, file)
                # Extract the folder number from the folder containing the .grd file
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                folder_number = get_number_from_folder(folder_name)
                sub_folder_name = os.path.basename(os.path.dirname(full_path))
                sub_folder_number = get_number_from_folder(sub_folder_name)
                if folder_number is not None:
                    grd_files.append((full_path, folder_number, sub_folder_number))

    # Sort the list by the folder number
    grd_files.sort(key=lambda x: x[2])
    grd_files.sort(key=lambda x: x[1])

    # Return only the file paths, not the folder numbers
    return [file_path for file_path, _, _ in grd_files]


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

def rot_M_theta(theta):
    theta_rad = np.radians(theta)  # Convert deg to rad
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad), np.cos(theta_rad), 0],
        [0, 0, 1]
    ])

def rot_M_psi(psi):
    psi_rad = np.radians(psi)  # Convert deg to rad
    return np.array([
        [np.cos(psi_rad), 0, np.sin(psi_rad)],
        [0, 1, 0],
        [-np.sin(psi_rad), 0, np.cos(psi_rad)]
    ])


def unrotate_spherical(coords, theta, psi):
    # return np.array([[np.matmul(rot_M_theta(-th), np.matmul(coords[i, j], rot_M_psi(-ps))) for j, ps in enumerate(psi)] for i, th in enumerate(theta)])
    return np.array([[np.matmul(rot_M_theta(-th), np.matmul(coords[i, j], rot_M_psi(-ps))) for j, ps in enumerate(psi)] for i, th in enumerate(theta)])


def fft_resample_complex_with_padding(array, new_shape):
    """
    Resample a 3D complex-valued numpy array along first 2 axis
    to a new shape using FFT resampling with edge padding.
    """
    # Pad the input array with constant values (default is 0)
    old_shape = array.shape
    padding = [(old_shape[i], old_shape[i]) if i < 2 else (0, 0) for i in range(len(array.shape))]
    padded_array = np.pad(array, padding, mode='edge')

    # Resample along each axis
    resampled = padded_array
    for axis, target_size in enumerate(new_shape[:2]):  # Only resample first 2 axes
        resampled = resample(resampled, target_size*3+2, axis=axis)

    # Crop or trim the resampled data to match the new shape in non-resized dimensions
    return resampled[new_shape[0]:2*new_shape[0], new_shape[1]:2*new_shape[1], :]

