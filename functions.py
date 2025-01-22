import os
import numpy as np
import re

from matplotlib import pyplot as plt
from scipy.signal import resample
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors


def spherical_to_cartesian(spherical_coords):
    if spherical_coords.ndim == 2:
        r = spherical_coords[:, 0]
        theta = spherical_coords[:, 1]  # azimuth angle (-180 to +180)
        psi = spherical_coords[:, 2]  # polar angle (-90 to +90)
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


def process_file(file_path, mask=None, polarisation=0):
    """
    :param polarisation: 0-vertical, 1 horizontal
    """
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
            real_part = float(numbers[0 + 2 * polarisation])
            imag_part = float(numbers[1 + 2 * polarisation])
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
    return np.array(
        [[np.matmul(rot_M_theta(-th), np.matmul(coords[i, j], rot_M_psi(-ps))) for j, ps in enumerate(psi)] for i, th in
         enumerate(theta)])


def rotate_point(coord, theta, psi):
    return np.array(
        [[np.matmul(rot_M_theta(th), np.matmul(coord, rot_M_psi(ps))) for j, ps in enumerate(psi)] for i, th in
         enumerate(theta)])


def fft_resample_complex_with_padding(array, new_shape):
    """
    Resample a 3D complex-valued numpy array along first 2 axis
    to a new shape using FFT resampling with edge padding.
    """
    # Pad the input array with constant values (default is 0)
    old_shape = array.shape
    padding = [(old_shape[i], old_shape[i]) if i < 2 else (0, 0) for i in range(len(array.shape))]
    padded_array = np.pad(array, padding, mode='symmetric')

    # Resample along each axis
    resampled = padded_array
    for axis, target_size in enumerate(new_shape[:2]):  # Only resample first 2 axes
        resampled = resample(resampled, target_size * 3 + 2, axis=axis)

    # Crop or trim the resampled data to match the new shape in non-resized dimensions
    return resampled[new_shape[0]:2 * new_shape[0], new_shape[1]:2 * new_shape[1], :]


def unwrap2D(array):
    uw0 = np.unwrap(array, axis=0)
    uw1 = np.unwrap(array, axis=1)
    if (uw0 == uw1).all():
        return uw0
    else:
        return uw0
        print("complicated unwrap")
        array = np.flip(array, axis=1)
        # plt.imshow(array, cflag='twilight')
        array[0, :] = np.unwrap(array[0, :])
        array[:, 0] = np.unwrap(array[:, 0])
        for i in range(1, len(array[:, 0])):
            for j in range(1, len(array[0, :])):
                tmp0 = np.abs(np.abs(array[i, j] - array[i - 1, j]) - np.pi)
                tmp1 = np.abs(np.abs(array[i, j] - array[i, j - 1]) - np.pi)
                if tmp0 < tmp1:
                    array[i, j] = np.unwrap((array[i, j - 1], array[i, j]))[1]
                else:
                    array[i, j] = np.unwrap((array[i - 1, j], array[i, j]))[1]
        # fig2 = plt.figure()
        # plt.imshow(array, cflag='twilight')
        # plt.colorbar()
        # plt.show()
        return np.flip(array, axis=1)


def connect_to_edge(point, array):
    """
    Connect a point to the nearest edge of the array with zeros.
    """
    n, m = array.shape
    edges = [(0, point[1]), (n - 1, point[1]), (point[0], 0), (point[0], m - 1)]
    distances = [np.abs(point[0] - e[0]) + np.abs(point[1] - e[1]) for e in edges]
    nearest_edge = edges[np.argmin(distances)]

    current_pos = np.array(point)
    array[tuple(current_pos)] = 0
    while not np.array_equal(current_pos, nearest_edge):
        diff = nearest_edge - current_pos
        step = np.sign(diff)
        current_pos += step
        array[tuple(current_pos)] = 0


def create_branch_cuts(input_array):
    """
    Connect the closest pairs of 1 and -1 with lines of zeros in the input array.
    """
    # Find positions of 1s and -1s

    one_array = np.ones([i+1 for i in input_array.shape])
    positions_1 = np.argwhere(input_array == 1)
    positions_minus1 = np.argwhere(input_array == -1)

    # Loop until all pairs are connected
    while len(positions_1) > 0 and len(positions_minus1) > 0:
        # Calculate Manhattan distances between all pairs
        distances = cdist(positions_1, positions_minus1, metric='cityblock')
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        p1 = positions_1[min_idx[0]]
        p2 = positions_minus1[min_idx[1]]

        # Draw line of zeros between p1 and p2
        current_pos = np.array(p1)
        one_array[tuple(current_pos)] = 0
        while not np.array_equal(current_pos, p2):
            diff = p2 - current_pos
            step = np.sign(diff)
            current_pos += step
            one_array[tuple(current_pos)] = 0

        # Remove connected points
        positions_1 = np.delete(positions_1, min_idx[0], axis=0)
        positions_minus1 = np.delete(positions_minus1, min_idx[1], axis=0)

    # Connect remaining 1s and -1s to the nearest edge
    for p in positions_1:
        connect_to_edge(p, one_array)
    for p in positions_minus1:
        connect_to_edge(p, one_array)

    return one_array


def unwrap_2d_branch_cut(phase):
    """
    Unwrap a 2D phase array using a branch-cut algorithm.
    """
    uw0 = np.unwrap(phase, axis=0)
    uw1 = np.unwrap(phase, axis=1)
    if (uw0 == uw1).all():
        return uw0

    diff_x = phase[:-1, :] - phase[1:, :]
    diff_y = phase[:, :-1] - phase[:, 1:]
    # Adjust phase differences to the range [-pi, pi]
    diff_x = (diff_x + np.pi) % (2 * np.pi) - np.pi
    diff_y = (diff_y + np.pi) % (2 * np.pi) - np.pi

    # Identify branch points
    residues = np.sign(np.round(+ diff_x[:, :-1] - diff_x[:, 1:] - diff_y[:-1, :] + diff_y[1:, :]))
    # residues = np.sign(np.round(+ diff_x[:, :-1] - diff_x[:, 1:] - diff_y[:-1, :] + diff_y[1:, :])) == 0


    flag = create_branch_cuts(residues)
    # Define the colormap and boundaries
    cmap0 = mcolors.ListedColormap(['red', 'white', 'blue'])  # Colors for -1, 0, and 1
    bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for the values
    norm0 = mcolors.BoundaryNorm(bounds, cmap0.N)
    cmap1 = mcolors.ListedColormap(['black', 'white'])  # Colors for -1, 0, and 1
    bounds = [-0.5, 0.5, 1.5]  # Boundaries for the values
    norm1 = mcolors.BoundaryNorm(bounds, cmap0.N)
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    c0 = axs[0].imshow(residues, cmap=cmap0, norm=norm0)
    fig.colorbar(c0, ax=axs[0], ticks=[-1, 0, 1])
    axs[0].set_title('Residues')
    c1 = axs[1].imshow(flag, cmap=cmap1, norm=norm1)
    axs[1].set_title('Branch cuts')
    # fig.colorbar(c1, ax=axs[1], ticks=[0, 1])
    plt.show()

    N, M = phase.shape
    flag[0, 0] = 2
    for l in range(5):
        for i in range(N):
            for j in range(M):
                if flag[i, j] == 2:
                    if i - 1 >= 0 and flag[i - 1, j] <= 1:
                        phase[i - 1, j] = unwrap_phase(phase[i, j], phase[i - 1, j])
                        flag[i - 1, j] = flag[i - 1, j] * 3 - 1
                    if i + 1 < N and flag[i + 1, j] <= 1:
                        phase[i + 1, j] = unwrap_phase(phase[i, j], phase[i + 1, j])
                        flag[i + 1, j] = flag[i + 1, j] * 3 - 1
                    if j - 1 >= 0 and flag[i, j - 1] <= 1:
                        phase[i, j - 1] = unwrap_phase(phase[i, j], phase[i, j - 1])
                        flag[i, j - 1] = flag[i, j - 1] * 3 - 1
                    if j + 1 < M and flag[i, j + 1] <= 1:
                        phase[i, j + 1] = unwrap_phase(phase[i, j], phase[i, j + 1])
                        flag[i, j + 1] = flag[i, j + 1] * 3 - 1
        if not 0 in flag:
            break
        fig, axs = plt.subplots(1, 2, figsize=(6, 6))
        axs[0].imshow(phase)
        axs[1].imshow(flag)
        plt.show()

    return phase


def unwrap_phase(reference, value):
    """
    Unwraps the value based on the reference by removing any 2π discontinuities.
    """
    return value + 2 * np.pi * np.round((reference - value) / (2 * np.pi))
