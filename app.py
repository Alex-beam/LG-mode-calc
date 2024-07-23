"""
Interactive program for online calculation of the profile of Laguerre-Gaussian resonator modes, local minima and maxima.
"""

import streamlit as st
import numpy as np
from scipy.special import eval_laguerre, roots_laguerre
from scipy.signal import argrelextrema
import pandas as pd

st.write(""" ## Laguerre-Gaussian Modes """)

p = st.sidebar.number_input('Select p index', 0, value = 5, max_value = 100)

r_max = st.sidebar.number_input(r'Select maximum r [$\mu $]', value = 800)
r_max /= 1e6
r = np.linspace(0, r_max, p*100 + 1000) # radius

w0 = st.sidebar.number_input(r'Select w0 [$\mu $] - radius', value = 200)
w0 /= 1e6             # [m] w0 - radius of gaussian mode

st.latex(r'''I = \frac{2}{\pi} \cdot \frac{1}{\omega^2_0} \cdot (L^0_p(\frac{2r^2}{\omega^2_0}))^2 
         \cdot exp(-\frac{2r^2}{\omega^2_0})''')



def I(p: int, r: np.ndarray[float], w0: float):
    """
    Electromagnetic field of Laguerre-Gaussian mode
    Parameters:
        p - radial index for modes with circular symmetry;
        r -  [m] is the radial distance from the center axis of the beam;
        w0 - [m] w0 - radius of gaussian mode;
    """

    X = 2*r**2 / w0**2
    Lp = eval_laguerre(p,X)
    return 2/np.pi * 1/(w0**2) * Lp**2 * np.exp(-X)
    

# Calculation and plot of Intensity
LG_pl = I(p, r, w0)
data = np.vstack((r*1e6, LG_pl/LG_pl[0]))
data = data.T
df = pd.DataFrame(data, columns = ['x', 'y'])
st.line_chart(df, x = 'x', x_label = 'r, um', y_label = 'Intensity')

# Finding local maxima
st.write(f"Local Maxima for {p = } in",  r"$\mu m$:")
if p > 0:
    i_max = argrelextrema(LG_pl,np.greater)
    r_max = [r[i] for i in i_max]
    s = "; ".join([str(np.round(x*1e6, 1)) for x in r_max[0]])  
    st.write(s)
else:
    st.write(f"Not found")

# Calculation of local minima LG modes
st.write(f"Local Minima for {p = } in",  r"$\mu m$:")
if p > 0:
    r_min = w0 * np.sqrt(roots_laguerre(p)[0] / 2)
    s = "; ".join([str((np.round(x*1e6, 1))) for x in r_min])  
    st.write(s)
else:
    st.write(f"Not found")

st.sidebar.write(""" Source code is located [here](https://github.com/Alex-beam/LG-mode-calc).   
    Contact me by [email](mailto:akorom@mail.ru).""")