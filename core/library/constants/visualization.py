"""Constants for plotting and visualization."""

import matplotlib.pyplot as plt
import matplotlib.cm as cm


MARKER_STYLES = [
    "o",  # Circle
    "^",  # Triangle Up
    "v",  # Triangle Down
    "<",  # Triangle Left
    ">",  # Triangle Right
    "p",  # Pentagon
    # "8",  # Octagon
    "D",  # Diamond
    "s",  # Square
    "d",  # Thin Diamond
    "h",  # Hexagon2
    "*",  # Star
    "X",  # X-shaped marker
    "P",  # Plus (filled)
    "x",  # X (cross)
    "H",  # Hexagon1
    # "|",  # Vertical Line
    # "_",  # Horizontal Line
    # ".",  # Point
    # ",",  # Pixel (small point)
    # "+",  # Plus
    # "h",  # Alternate Hexagon
    # "H",  # Alternate Hexagon (filled)
    # "p",  # Alternate Pentagon
    # "*",  # Alternate Star
    # "X",  # Alternate X
    # "1",  # Tri-down (Alternative)
    # "2",  # Tri-up (Alternative)
    # "3",  # Tri-left (Alternative)
    # "4",  # Tri-right (Alternative)
]


# MARKER_COLORS = list(plt.cm.tab40.colors)
# Get the default color cycle
#
# DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# DEFAULT_COLORS = plt.cm.tab20.colors

# num_colors = 20
# colormap = cm.get_cmap("turbo", num_colors)  # 'viridis' has good contrast #turbo
# DEFAULT_COLORS = [colormap(i) for i in range(num_colors)]

DEFAULT_COLORS = [
    "blue",
    "red",
    "magenta",
    "darkgoldenrod",  # Dark yellow option
    "green",
    "black",
    "orange",
    "purple",
    "brown",
    "darkslategray",
    "indigo",
    "gray",
    # "cyan",
    # "pink",
    # "lime",
    # "teal",
    # "olive",
    # "maroon",
    # "gold",
    # "navy",
    # "darkred",        # Added
    # "darkviolet"      # Added
    # "darkorange",     # Added
    "darkgreen",
    "violet",
]
