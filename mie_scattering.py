from miepython import mie
import numpy as np
import matplotlib.pyplot as plt

c = 299792458.
epsilon_0 = 8.854e-12
pi = np.pi

start = 0.002
stop = 0.006
n = 17


def PEC_rcs(pec_radius, freq):
    # Define the radius and frequency
    radius = pec_radius  # Radius of the PEC ball
    k = 2 * pi * freq / c   # Calculate the Wavenumber

    # Complex refractive index for PEC
    m = 1e-3-1j*1e+5

    # Calculate the Mie coefficients for scattering
    qext, qsca, qback, g = mie(m, radius*k)

    # Calculate the RCS using the Mie coefficients
    rcs = pi * radius**2 * qback
    return rcs


plt.plot(np.linspace(start, stop, n, True), [PEC_rcs(r, 24e9)/(r**2*np.pi) for r in np.linspace(start, stop, n, True)])
plt.show()

print([PEC_rcs(r, 24e9)/(r**2*np.pi) for r in np.linspace(start, stop, n, True)])
print([r for r in np.linspace(start, stop, n, True)])


