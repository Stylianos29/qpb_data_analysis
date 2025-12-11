import numpy as np
import matplotlib.pyplot as plt
import gvar as gv

# Data
x = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
)
y_mean = np.array(
    [
        3.2997,
        2.1353,
        1.850,
        1.730,
        1.657,
        1.610,
        1.576,
        1.551,
        1.541,
        1.534,
        1.527,
        1.521,
        1.513,
        1.509,
        1.508,
        1.513,
        1.525,
        1.552,
        1.616,
        1.723,
        1.861,
        2.007,
    ]
)
y_error = np.array(
    [
        0.00459794,
        0.00754023,
        0.01124955,
        0.01666036,
        0.02104568,
        0.02663433,
        0.03122901,
        0.03588413,
        0.04005118,
        0.04320726,
        0.04495486,
        0.04671572,
        0.04727865,
        0.04901289,
        0.05076231,
        0.053096,
        0.0553041,
        0.05813505,
        0.06011091,
        0.06248843,
        0.06369647,
        0.06465021,
    ]
)

# Convert to gvar objects
y = gv.gvar(y_mean, y_error)
# Calculate numerical derivative
dy_dx_forward = np.diff(y) / np.diff(x)
# print(len(dy_dx_forward), len(y))
print(dy_dx_forward)

# Check for plateau: |dy_dx_forward.mean| < dy_dx_forward.sdev (1 sigma condition)
plateau_indices = [
    i for i, dy in enumerate(dy_dx_forward) if abs(dy.mean) <= 0.5 * dy.sdev
]

dy_dx_centered = (y[2:] - y[:-2]) / 2

print(plateau_indices)

# dy_dx_backward = -np.diff(y[::-1])[::-1]
print(dy_dx_centered)
plateau_indices_centered = [
    i + 1 for i, dy in enumerate(dy_dx_centered) if abs(dy.mean) <= 0.5 * dy.sdev
]

print(plateau_indices_centered)

y_plateau = y[plateau_indices_centered]
y_mean_plateau = gv.mean(y_plateau)
y_error_plateau = gv.sdev(y_plateau)

# Weights: inverse variance
weights = 1 / (y_error_plateau**2)

# Weighted mean (plateau value)
plateau_value = np.sum(weights * y_mean_plateau) / np.sum(weights)

# Error of the flat line
plateau_error = np.sqrt(len(weights) / 3) * np.sqrt(1 / np.sum(weights))

# Output results
print(f"Optimal plateau value: {plateau_value:.5f}")
print(f"Plateau error: {plateau_error:.5f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt="o", label="Data with Errors")
plt.axhline(
    plateau_value,
    color="red",
    linestyle="--",
    label=f"Plateau: {plateau_value:.5f} ± {plateau_error:.5f}",
)
plt.fill_between(
    x,
    plateau_value - plateau_error,
    plateau_value + plateau_error,
    color="r",
    alpha=0.2,
)
plt.axvspan(
    x[plateau_indices_centered[0]],
    x[plateau_indices_centered[-1]],
    color="green",
    alpha=0.3,
    label="Plateau Range",
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Optimal Plateau Line with Error")
plt.legend()
# plt.show()


# print(plateau_indices)
# plateau_x = plateau_indices
# # Group contiguous indices to determine ranges
# plateau_ranges = []
# start = None
# for idx in plateau_indices:
#     if start is None:
#         start = idx
#     elif idx != start + len(plateau_ranges):
#         plateau_ranges.append((start, idx))
#         start = idx
# if start is not None:
#     plateau_ranges.append((start, plateau_indices[-1] + 1))

# # Select the largest plateau range
# if plateau_ranges:
#     largest_plateau = max(plateau_ranges, key=lambda r: r[1] - r[0])
#     plateau_x = x[largest_plateau[0]:largest_plateau[1] + 1]
#     print(f"Plateau likely in the range: {plateau_x[0]} to {plateau_x[-1]}")
# else:
#     print("No plateau found.")

# # Visualization
# plt.figure(figsize=(10, 6))
# plt.errorbar(x, gv.mean(y), yerr=gv.sdev(y), fmt='o', label='Data with Errors')
# plt.axvspan(plateau_x[0], plateau_x[-1], color='green', alpha=0.3, label='Detected Plateau')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plateau Detection with gvar')
# plt.legend()
# # plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import gvar as gv

# # Data
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
# y_mean = np.array([3.2997, 2.1353, 1.850, 1.730, 1.657, 1.610, 1.576,
#               1.551, 1.541, 1.534, 1.527, 1.521, 1.513, 1.509,
#               1.508, 1.513, 1.525, 1.552, 1.616, 1.723, 1.861, 2.007])
# y_error = np.array([0.00459794, 0.00754023, 0.01124955, 0.01666036, 0.02104568, 0.02663433,
#                     0.03122901, 0.03588413, 0.04005118, 0.04320726, 0.04495486, 0.04671572,
#                     0.04727865, 0.04901289, 0.05076231, 0.053096, 0.0553041, 0.05813505,
#                     0.06011091, 0.06248843, 0.06369647, 0.06465021])

# y = gv.gvar(y_mean, y_error)

# # Calculate numerical derivative
# # dy_dx_forward = gv.abs(np.diff(y)) / np.diff(x)


# dy_dx_forward = np.diff(y) / np.diff(x)

# # Sliding window standard deviation (include y_error)
# window_size = 5
# std_devs = np.array([np.std(y[i:i+window_size]) + np.mean(y_error[i:i+window_size])
#                      for i in range(len(y) - window_size + 1)])
# x_windows = x[:len(std_devs)]

# # Truncate dy_dx_forward to match std_devs length
# dy_dx_forward = dy_dx_forward[:len(std_devs)]

# # Identify plateau regions with preference for larger ranges
# threshold = 0.1
# flat_regions = (dy_dx_forward < threshold) & (std_devs < threshold)
# plateau_ranges = []
# start = None

# for i, is_flat in enumerate(flat_regions):
#     if is_flat:
#         if start is None:
#             start = i
#     elif start is not None:
#         plateau_ranges.append((start, i))
#         start = None

# if start is not None:
#     plateau_ranges.append((start, len(flat_regions)))

# # Select the largest plateau range
# if plateau_ranges:
#     largest_plateau = max(plateau_ranges, key=lambda r: r[1] - r[0])
#     plateau_x = x_windows[largest_plateau[0]:largest_plateau[1] + 1]
#     print(f"Plateau likely in the range: {plateau_x[0]} to {plateau_x[-1]}")
# else:
#     print("No plateau found.")

# # Plot data and indicators
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, label='Data', marker='o')
# plt.plot(x[1:1+len(dy_dx_forward)], dy_dx_forward, label='|dy/dx|', marker='x')
# plt.plot(x_windows, std_devs, label='Sliding Std Dev (with errors)', marker='s')
# plt.axhline(y=0, color='r', linestyle='--', label='Threshold (Flatness)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Identifying Plateau in Data with Errors')
# # plt.show()


plt.savefig("TEST.png")
plt.close()

# y = gv.gvar(y_mean, y_error)
# print(np.diff(y)/ np.diff(x))
