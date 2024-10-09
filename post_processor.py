import numpy as np

dx = 1.6*2/100
dy = 1*2/100
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


# Example usage
file_path = r"Ticra_results/try_ticra_out.txt.grd"


received = process_file(file_path)
print(received)

zero_phase = np.angle(np.exp(1j*r**2*k))
phase_center = np.array(((np.angle(received[0]) + zero_phase) / k / 2,
                         dx / 2 + r / dx / k * (np.angle(received[1]) - np.angle(received[0])),
                         dy / 2 + r / dy / k * (np.angle(received[2]) - np.angle(received[0]))))
print(phase_center*100)
