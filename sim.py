import random

import numpy as np
from matplotlib import pyplot as plt
import csv

c = 299792458  # m/s
d = 0.015  # distance between receivers
# d = 0.03  # distance between receivers
r = 2  # distance to target


def rot_matrix(rot, deg=True):
    if deg:
        rot = np.deg2rad(rot)
    return np.array([[np.cos(rot), -np.sin(rot), 0], [np.sin(rot), np.cos(rot), 0], [0, 0, 1]])


def spherical_to_cartesian(spherical_coords):
    r = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]  # azimuthal angle (-180 to +180)
    phi = spherical_coords[:, 2]    # polar angle (-90 to +90)
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta) + d / 2
    z = r * np.sin(phi) + d / 2
    cartesian_coords = np.vstack((x, y, z)).T
    return cartesian_coords


class PointScatterer:
    def __init__(self, RCS_dB, x, y, z):
        self.RCS = 10**(RCS_dB/10)
        self.pos = np.array([x, y, z])

    def bcscatter(self, source, recpos, rot, deg=True):

        pos = np.matmul(rot_matrix(rot, deg), self.pos)
        r1 = np.linalg.norm(pos-source.pos)
        r2 = np.linalg.norm(pos-recpos)

        return np.exp(-1j*(r1+r2)*source.k) * np.sqrt(self.RCS) / r1 / r2


class Source:
    def __init__(self, f, x, y, z):
        self.pos = np.array([x, y, z])
        self.f = f
        self.lamb = c/f
        self.k = 2 * np.pi / self.lamb


def find_best_fit_phase_center(rot_angles, phase_center):
    unrotated_ph_c = np.array([np.matmul(rot_matrix(-rot, True), phase_center[i, :])
                               for i, rot in enumerate(rot_angles)])
    ph_c_fit = np.average(unrotated_ph_c, axis=0)
    fitted_phase_center = np.array([np.matmul(rot_matrix(rot, True), ph_c_fit)
                                    for rot in rot_angles])
    ph_c_variance = phase_center - fitted_phase_center
    return ph_c_fit, ph_c_variance


source = Source(24E9, -2, 0, 0)
rec_poses = np.array([[-r, 0, 0], [-r, d, 0], [-r, 0, d]])
point_scatterers = [PointScatterer(0, 0, 0.0, 0)]
# point_scatterers = []
#
#
for i in range(20):
    point_scatterers.append(PointScatterer(random.gauss(-20, 3),
                                           random.gauss(0, 0.02),
                                           random.gauss(0, 0.02),
                                           0))

rot_angles = np.linspace(0, 360, 3600, endpoint=False)
received = np.zeros((len(rot_angles), len(rec_poses)), dtype=complex)

for i, rot in enumerate(rot_angles):
    rec = np.array([0. + 0j for r_pos in rec_poses])
    for ps in point_scatterers:
        rec += np.array([ps.bcscatter(source, r_pos, rot, deg=True) for r_pos in rec_poses])
    received[i, :] = rec

zero_phase = np.angle(np.exp(1j*r**2*source.k))

phase_center_sph = np.array((-np.unwrap(np.angle(received[:, 0]) + zero_phase)/source.k/2 + r,
                             np.arcsin(np.unwrap(np.angle(received[:, 1]) - np.angle(received[:, 0])) / source.k / d),
                             np.arcsin(np.unwrap(np.angle(received[:, 2]) - np.angle(received[:, 0])) / source.k / d))).T

phase_center = spherical_to_cartesian(phase_center_sph) + np.array((-r, 0, 0))

best_fit_ph_c, ph_c_err = find_best_fit_phase_center(rot_angles, phase_center)
print("Best fit for the phase center is: ", best_fit_ph_c)

# fig_ph, axs = plt.subplots(1, figsize=(9, 4))
#
# axs.plot(rot_angles,  np.rad2deg(np.unwrap(np.angle(recived[:, 0]) - np.angle(recived[:, 1]))), 'g')
# # axs.plot(rot_angles, np.rad2deg(np.unwrap(np.angle(recived[:, 1]))), 'r')
# axs.set_xlabel('Rotation Angle [degrees]')
# axs.set_ylabel('Angle Difference [degrees]')
# axs.grid()

fig, axs = plt.subplots(4, figsize=(9, 9))

# x
axs[0].set_title('X Coordinate')
axs[0].plot(rot_angles, phase_center[:, 0] * 100, 'g')
axs[0].plot(rot_angles, ph_c_err[:, 0] * 100, 'r')
axs[0].set_xlabel('Rotation Angle [degrees]')
axs[0].set_ylabel('Phase Center X [cm]')
axs[0].grid()

# y
axs[1].set_title('Y Coordinate')
axs[1].plot(rot_angles, phase_center[:, 1] * 100, 'g')
axs[1].plot(rot_angles, ph_c_err[:, 1] * 100, 'r')
axs[1].set_xlabel('Rotation Angle [degrees]')
axs[1].set_ylabel('Phase Center Y [cm]')
axs[1].grid()

# y
axs[2].set_title('Y Coordinate')
axs[2].plot(rot_angles, phase_center[:, 2] * 100, 'g')
axs[2].plot(rot_angles, ph_c_err[:, 2] * 100, 'r')
axs[2].set_xlabel('Rotation Angle [degrees]')
axs[2].set_ylabel('Phase Center Z [cm]')
axs[2].grid()


# power
p_max = np.max(np.abs(received[:, 0]))
axs[3].set_title('Normalised received power')
axs[3].plot(rot_angles, 20 * np.log10(np.abs(received[:, 0])/p_max), 'r')
axs[3].set_xlabel('Rotation Angle [degrees]')
axs[3].set_ylabel('Power [dB]')
axs[3].grid()

# Adding a common figure label for phase center
fig.suptitle('Phase Center Coordinates', fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the super title


# Start of your plotting code
fig2, axs = plt.subplots(1, figsize=(6, 6))

# Extract value, x, and y columns
values = np.array([ps.RCS for ps in point_scatterers])
x = np.array([ps.pos[0] for ps in point_scatterers])
y = np.array([ps.pos[1] for ps in point_scatterers])

# Create the scatter plot
axs.scatter(x * 100, y * 100, color='royalblue', s=values * 5000, alpha=0.5)
axs.scatter(best_fit_ph_c[0] * 100, best_fit_ph_c[1] * 100, color='r', label='Best fit phase center')

plt.xlabel('X [cm]')
plt.ylabel('Y [cm]')
axs.set_ylim([-6, 6])
axs.set_xlim([-6, 6])  # Set the x-axis limits
axs.set_aspect('equal')  # Ensure equal aspect ratio

# Example sizes for the legend (you can adjust these)
sizes = [0.01 * 5000, 10**-1.5 * 5000, 0.1 * 5000]  # Scale factor of 5000 applied

# Customize the legend
handles, labels = axs.get_legend_handles_labels()
legend_circles = [plt.Line2D([0], [0], marker='o', linestyle='none', color='royalblue', alpha=0.5,
                             markersize=np.sqrt(size))
                  for size in sizes]
legend_texts = ['Point scatterer, RCS = -20 [dB]', 'Point scatterer, RCS = -15 [dB]', 'Point scatterer, RCS = -10 [dB]']

plt.legend(handles + legend_circles,
           labels + legend_texts, frameon=True)

axs.grid()

plt.show()

data = np.column_stack((np.real(rot_angles),
                        np.real(phase_center[:, 0])*100,
                        np.real(phase_center[:, 1])*100,
                        np.real(ph_c_err[:, 0])*100,
                        np.real(ph_c_err[:, 1])*100,
                        20 * np.log10(np.sum(np.abs(received), axis=1)/p_max)))

# Column names
columns = ['rot_angles', 'ph_c_x', 'ph_c_y', 'ph_c_err_x', 'ph_c_err_y', 'power']

# Save to CSV
with open('PS_sim_single_rotating.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(columns)  # Write column headers
    writer.writerows(data)    # Write data rows






