import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os
import sys
import scienceplots

import matplotlib.pyplot as plt
print(plt.style.available)

plt.style.use(['science','ieee'])


## use Myriad Pro front
# Step 1: Absolute path to local Myriad Pro TTF

FONT_FILE = f"{os.path.dirname(os.path.abspath(__file__))}/MyriadPro-Regular.ttf"

# FONT_FILE = f"{os.path.dirname(os.path.abspath(__file__))}/Caprasimo-Regular.ttf"

if not os.path.isfile(FONT_FILE):
    sys.exit(f"{__file__}: Font file not found at: {FONT_FILE}!")

# Step 2: Register the font with Matplotlib
myriad_font = fm.FontProperties(fname=FONT_FILE)
font_name = myriad_font.get_name()

# Add it to Matplotlib's font manager
fm.fontManager.addfont(FONT_FILE)

# Step 3: Set as default font globally for this session
mpl.rcParams['font.family'] = font_name


plt.rcParams.update({
    'font.family': 'Myriad Pro',
    # 'font.size': 24,
    # 'axes.linewidth': 1.2,
    'text.usetex': False,
    'mathtext.default': 'regular',
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
})


if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plt.grid()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Myriad Pro as Default")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.tight_layout()
    plt.show()