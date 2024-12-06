from mpl_toolkits import mplot3d
from stl import mesh
import numpy as np
from matplotlib import pyplot as plt
from functions import spherical_to_cartesian, process_file, get_grd_files_grid, unrotate_spherical, fft_resample_complex_with_padding
from scipy.interpolate import RegularGridInterpolator

# Constants
# d = 1.5/100*2  # 1.5 cm receiver spacing
d = 6/100  # 6 cm receiver spacing
# d = 20/100  # 20 cm receiver spacing
r = 2  # m
f = 24e9  # Hz
# f = 10e9  # Hz
c = 299792458  # m/s
k = f*2*np.pi/c  # rad/m
E_i = 0.1138130187E-02 - 0.9662416040E-03j  # incident field (r=2 m) 24GHz
# E_i = -0.761989360E-03 - 0.3493495524E-02j  # incident field (r=2 m) 10GHz

target_grid = (41, 21)


text = ""


def process_folder_grid(base_path):
    grd_paths = get_grd_files_grid(base_path)
    sig = []
    for grd in grd_paths:
        sig.append(process_file(grd, mask=[-9, -7, -3]))  # use for 3*3 grid
        # sig.append(process_file(grd, mask=[-5, -4, -2]))  # use for 3*3 grid
        # sig.append(process_file(grd, mask=[-4, -3, -2]))  # use for 2*2 grid

    return np.array(sig)


# Example usage
# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\f10GHz_80_100_85_95"
# cad_model = mesh.Mesh.from_file('Fill_all_gap_V4.STL')

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Try_80_100_85_95"
folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\Try_80_100_85_95_6cm_spacing"

cad_model = mesh.Mesh.from_file('Fill_all_gap_V4.STL')

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\ugly_ducling_80_100_85_95"
# cad_model = mesh.Mesh.from_file('ugly_ducling.STL')

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\ugly_ducling_jc_80_100_85_95"
# cad_model = mesh.Mesh.from_file('ugly_ducling_jc.STL')

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\ugly_ducling_js_80_100_85_95"
# cad_model = mesh.Mesh.from_file('ugly_ducling_js.STL')

radial_offset = 0.08

# folder_path = r"C:\Users\titiv\OneDrive - Danmarks Tekniske Universitet\DTU\Thesis\sim_results\double_corner_30_30deg"

received = process_folder_grid(folder_path)
received = np.flip(received.reshape(21, 11, 3), 1)
theta = np.linspace(-10, +10, 41, True)
psi = np.linspace(+5, -5, 21, True)

received = fft_resample_complex_with_padding(received, (41, 21, 3))


# received = np.flip(received.reshape(11, 5, 3), 1)
# theta = np.linspace(-10, +10, 11, True)
# psi = np.linspace(+5, -5, 5, True)

# received = received.reshape(7, 7, 3)
# theta = np.linspace(-30, 0, 7, True)
# psi = np.linspace(-30, 0, 7, True)


zero_phase = np.angle(np.exp(1j*(r-radial_offset)**2*k))

r_ = -np.unwrap(np.unwrap(np.angle(received[:, :, 0]) + zero_phase, axis=0), axis=1)/k/2 + r - radial_offset
# theta_ = np.arcsin(np.unwrap(np.unwrap(((np.angle(received[:, :, 1]) - np.angle(received[:, :, 0]) + np.pi) %
#                                         (2 * np.pi) - np.pi), axis=1), axis=0) / k / d)
# theta_ = np.arcsin(np.unwrap(np.unwrap(np.angle(received[:, :, 1]), axis=1), axis=0) - np.unwrap(np.unwrap(np.angle(received[:, :, 0]), axis=1), axis=0) / k / d)
theta_ = np.arcsin(np.unwrap(np.unwrap(((np.angle(received[:, :, 1]) - np.angle(received[:, :, 0]) + np.pi) %
                                       (2 * np.pi) - np.pi), axis=1), axis=0) / k / d)
psi_ = np.arcsin(np.unwrap(np.unwrap(((np.angle(received[:, :, 2]) - np.angle(received[:, :, 0]) + np.pi) %
                                      (2 * np.pi) - np.pi), axis=1), axis=0) / k / d)

phase_center_sph = np.stack((r_, theta_, psi_), axis=-1)
phase_center = spherical_to_cartesian(phase_center_sph) #+ np.array((0, d/2, d/2))


# #figure used for plotting results from reflectors
# fig, axes = plt.subplots(2, 2, figsize=(8, 6))
#
# # Plot x coordinates
# # c1 = axes[0, 0].imshow(phase_center[:, :, 0] * 100 - 200, extent=[psi.min(), psi.max(), theta.max(), theta.min()],
# #                        origin='upper', aspect='equal', cmap='viridis')
# # axes[0, 0].set_title('x')
# # axes[0, 0].set_ylabel('Theta [deg]', labelpad=-0)
# # axes[0, 0].set_xlabel('Psi [deg]')
# # fig.colorbar(c1, ax=axes[0, 0], label='[cm]')
#
# c1 = axes[0, 1].imshow((phase_center_sph[:, :, 0] - r + radial_offset) * 100, extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                        origin='upper', aspect='equal', cmap='viridis')
# axes[0, 1].set_title(r"$\Delta r$")
# axes[0, 1].set_ylabel('Theta [deg]', labelpad=-0)
# axes[0, 1].set_xlabel('Psi [deg]')
# fig.colorbar(c1, ax=axes[0, 1], label='[cm]')
#
# # Plot RCS (in dB)
# c2 = axes[0, 0].imshow(20 * np.log10(np.abs(received[:, :, 0] / E_i * r * 2 * np.sqrt(np.pi))),
#                        extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                        origin='upper', aspect='equal', cmap='viridis')
# axes[0, 0].set_title('RCS')
# axes[0, 0].set_ylabel('Theta [deg]', labelpad=-0)
# axes[0, 0].set_xlabel('Psi [deg]')
# fig.colorbar(c2, ax=axes[0, 0], label='[dBsm]')
#
# plt.tight_layout()
#
# # Plot y coordinates
# c3 = axes[1, 0].imshow(phase_center[:, :, 1] * 100, extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                        origin='upper', aspect='equal', cmap='viridis')
# axes[1, 0].set_title('y')
# axes[1, 0].set_ylabel('Theta [deg]', labelpad=-0)
# axes[1, 0].set_xlabel('Psi [deg]')
# fig.colorbar(c3, ax=axes[1, 0], label='[cm]')
#
# # Plot z coordinates
# c4 = axes[1, 1].imshow(phase_center[:, :, 2] * 100, extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                        origin='upper', aspect='equal', cmap='viridis')
# axes[1, 1].set_title('z')
# axes[1, 1].set_ylabel('Theta [deg]', labelpad=-0)
# axes[1, 1].set_xlabel('Psi [deg]')
# fig.colorbar(c4, ax=axes[1, 1], label='[cm]')
#
# plt.tight_layout()
# plt.show()

# #figure used for plotting results from club
fig, axes = plt.subplots(1, 4, figsize=(10, 3))

# Plot x coordinates (delta r)
c1 = axes[1].imshow((phase_center_sph[:, :, 0] - r + radial_offset) * 100,
                    extent=[psi.min(), psi.max(), theta.max(), theta.min()],
                    origin='upper', aspect='equal', cmap='viridis')
axes[1].set_title(r"$\Delta r$")
axes[1].set_ylabel('Theta [deg]', labelpad=-0)
axes[1].set_xlabel('Psi [deg]')
fig.colorbar(c1, ax=axes[1], label='[cm]')

# Plot RCS (in dB)
c2 = axes[0].imshow(20 * np.log10(np.abs(received[:, :, 0] / E_i * r * 2 * np.sqrt(np.pi))),
                    extent=[psi.min(), psi.max(), theta.max(), theta.min()],
                    origin='upper', aspect='equal', cmap='viridis')
axes[0].set_title('RCS')
axes[0].set_ylabel('Theta [deg]', labelpad=-0)
axes[0].set_xlabel('Psi [deg]')
fig.colorbar(c2, ax=axes[0], label='[dBsm]')

# Plot y coordinates
c3 = axes[2].imshow(phase_center[:, :, 1] * 100,
                    extent=[psi.min(), psi.max(), theta.max(), theta.min()],
                    origin='upper', aspect='equal', cmap='viridis')
axes[2].set_title('y')
axes[2].set_ylabel('Theta [deg]', labelpad=-0)
axes[2].set_xlabel('Psi [deg]')
fig.colorbar(c3, ax=axes[2], label='[cm]')

# Plot z coordinates
c4 = axes[3].imshow(phase_center[:, :, 2] * 100,
                    extent=[psi.min(), psi.max(), theta.max(), theta.min()],
                    origin='upper', aspect='equal', cmap='viridis')
axes[3].set_title('z')
axes[3].set_ylabel('Theta [deg]', labelpad=-0)
axes[3].set_xlabel('Psi [deg]')
fig.colorbar(c4, ax=axes[3], label='[cm]')

plt.tight_layout()
plt.show()



#
# fig2, axes = plt.subplots(1, 4, figsize=(18, 7))
#
# # Plot x coordinates (angle of received[:, :, 0])
# # c1 = axes[0].imshow(np.angle(received[:, :, 0]), extent=[psi.min(), psi.max(), theta.max(), theta.min()],
# c1 = axes[0].imshow(np.unwrap(np.unwrap(np.angle(received[:, :, 0]), axis=1), axis=0), extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                     aspect='equal', cmap='twilight')
# axes[0].set_title('r [cm]')
# axes[0].set_ylabel('Theta [deg]')
# axes[0].set_xlabel('Psi [deg]')
# fig2.colorbar(c1, ax=axes[0], label='Phase [rad]')
#
# # Plot y coordinates (angle difference between received[:, :, 1] and received[:, :, 0])
# # c2 = axes[1].imshow((np.angle(received[:, :, 1]) - np.angle(received[:, :, 0]) + np.pi) % (2 * np.pi) - np.pi,
# c2 = axes[1].imshow(np.unwrap(np.unwrap(np.angle(received[:, :, 1]), axis=1), axis=0) -
#                     np.unwrap(np.unwrap(np.angle(received[:, :, 0]), axis=1), axis=0),
#                     extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                     aspect='equal', cmap='twilight')
# axes[1].set_title('Azimuth [deg]')
# axes[1].set_ylabel('Theta [deg]')
# axes[1].set_xlabel('Psi [deg]')
# fig2.colorbar(c2, ax=axes[1], label='Phase Difference [rad]')
#
# # Plot z coordinates (angle difference between received[:, :, 2] and received[:, :, 0])
# # c3 = axes[2].imshow((np.angle(received[:, :, 2]) - np.angle(received[:, :, 0]) + np.pi) % (2 * np.pi) - np.pi,
# c3 = axes[2].imshow(np.unwrap(np.unwrap(np.angle(received[:, :, 1]), axis=1), axis=0),
#
#                     extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                     aspect='equal', cmap='twilight')
# axes[2].set_title('Polar [deg]')
# axes[2].set_ylabel('Theta [deg]')
# axes[2].set_xlabel('Psi [deg]')
# fig2.colorbar(c3, ax=axes[2], label='Phase Difference [rad]')
#
# # Plot RCS (in dB)
# c4 = axes[3].imshow(20 * np.log10(np.abs(received[:, :, 0] / E_i * r * 2 * np.sqrt(np.pi))),
#                     extent=[psi.min(), psi.max(), theta.max(), theta.min()],
#                      aspect='equal', cmap='viridis')
# axes[3].set_title('RCS [dBsm]')
# axes[3].set_ylabel('Theta [deg]')
# axes[3].set_xlabel('Psi [deg]')
# fig2.colorbar(c4, ax=axes[3], label='RCS [dBsm]')
#
# plt.tight_layout()

# unrotated_phase_center = unrotate_spherical(phase_center - np.array((r, 0, 0)), theta, psi)
#
# x = unrotated_phase_center[:, :, 0].flatten()
# y = unrotated_phase_center[:, :, 1].flatten()
# z = unrotated_phase_center[:, :, 2].flatten()
#
# # Cad models used in the simulation and the one used for the plotting era offset because of the different file format
# x_ = x + 0.1071479
# y_ = y + 0.02978903
# z_ = z + 0.07139254
#
# # Calculate the color values
# color_values = 20 * np.log10(np.abs(received[:, :, 0] / E_i * r * 2 * np.sqrt(np.pi)))
# color_values_flat = color_values.flatten()
#
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the STL model
# poly_collection = mplot3d.art3d.Poly3DCollection(cad_model.vectors)
# poly_collection.set_color((0.7, 0.2, 0.2, 0.0))  # Adjust color if needed
# poly_collection.set_edgecolor((0.7, 0.2, 0.2, 0.1))  # Half-transparent edges
# ax.add_collection3d(poly_collection)
# # Plot the point cloud
# sc = ax.scatter(x_, y_, z_, c=color_values_flat, cmap='viridis', marker='.')
# # Add a color bar
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('RCS [dBsm]')
# # Set labels
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.set_title('3D Point Cloud from phase_center Data with CAD Model')
# # Set equal axis limits
# ax.set_xlim(-0.01, 0.1)
# ax.set_ylim(-0.01, 0.1)
# ax.set_zlim(-0.01, 0.1)
# # Set the view direction: Looking from (1, 0, 0) with z-axis pointing left
# ax.view_init(elev=0, azim=180, vertical_axis='y')  # Adjust for x-axis as the main viewing direction

plt.show()
