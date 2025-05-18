import streamlit as st
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from control import tf, bode_plot, nyquist_plot, step_response, pzmap, root_locus, poles




st.set_page_config(layout="wide")
st.title("Jonas' Interaktives Regelungstechnik-Tool")

# Sidebar Tabs
input_mode = st.sidebar.radio("Systemeingabe-Modus:", ["Übertragungsfunktion", "Pole/Nullstellen"])

# Frequenzbereich für Bode/NY
w = np.logspace(-2, 2, 1000)

def parse_complex_list(s):
    s = s.replace(" ", "")
    result = []
    if not s:
        return result  # leere Liste bei leerem Feld
    for part in s.split(","):
        if part:  # nur wenn part nicht leer ist
            try:
                result.append(complex(part))
            except ValueError:
                result.append(float(part))  # Fallback
    return result


# Ersatz für tf(signal.zpk2tf(...))



# ---- Systemeingabe ---- #
if input_mode == "Übertragungsfunktion":
    num_str = st.sidebar.text_input("Zählerkoeffizienten (z.B. 1, 3):", "1")
    den_str = st.sidebar.text_input("Nennerkoeffizienten (z.B. 1, 2, 1):", "1, 2, 1")
    try:
        num = parse_complex_list(num_str)
        den = parse_complex_list(den_str)

        plant = tf(num, den)
    except Exception as e:
        st.error(f"Fehler beim Parsen der Übertragungsfunktion: {e}")
        plant = tf([1], [1])
elif input_mode == "Pole/Nullstellen":
    zeros_str = st.sidebar.text_input("Nullstellen (z.B. -1, -2):", "-1")
    poles_str = st.sidebar.text_input("Pole (z.B. -1, -3):", "-1, -3")
    gain = 1
    try:
        z = parse_complex_list(zeros_str)
        p = parse_complex_list(poles_str)
        num, den = signal.zpk2tf(z, p, gain)
        plant = tf(num, den)
    except Exception as e:
        st.error(f"Fehler beim Parsen von Nullstellen/Polen: {e}")
        plant = tf([1], [1])

# ---- Reglerauswahl ---- #
st.sidebar.markdown("---")
st.sidebar.header("Regler")
regler_typ = st.sidebar.selectbox("Reglertyp", ["P", "PI", "PD"])
Kp = st.sidebar.slider("Kp (Verstärkung)", 0.0, 10.0, 1.0)
Ki = Kd = 0.0

if regler_typ == "PI":
    Ki = st.sidebar.slider("Ki (Integralanteil)", 0.0, 10.0, 1.0)
    controller = tf([Kp * Ki, Kp], [1, 0])
elif regler_typ == "PD":
    Kd = st.sidebar.slider("Kd (Differentialanteil)", 0.0, 5.0, 0.5)
    controller = tf([Kd, Kp], [1])
else:
    controller = tf([Kp], [1])


# ---- Lead/Lag ---- #
st.sidebar.markdown("---")
st.sidebar.header("Lead/Lag-Element")
lead_lag_enable = st.sidebar.checkbox("Aktivieren")
if lead_lag_enable:
    z = st.sidebar.slider("z (Lead-Nullstelle, 1/z = Position)", 0.1, 10.0, 1.0)
    p = st.sidebar.slider("p (Lead-Pol, 1/p = Position)", 0.1, 10.0, 2.0)
    leadlag = tf([z, 1], [p, 1])
    
    st.sidebar.latex(r"\text{Lead/Lag: } \frac{1 + z s}{1 + p s}")
    st.sidebar.markdown(f"mit $z = {z}$, $p = {p}$")
else:
    leadlag = tf([1], [1])
    z = p = None  # Falls später geprüft


# ---- Gesamtsystem ---- #
open_loop = controller * plant * leadlag
closed_loop = open_loop / (1 + open_loop)

# Stabilitätsprüfung
pole_list = poles(closed_loop)
is_stable = all(np.real(p) < 0 for p in pole_list)

# Schwingfähigkeit prüfen: Komplexe konjugierte Pole mit negativem Realteil
is_oscillatory = any(np.real(p) < 0 and np.imag(p) != 0 for p in pole_list)


st.markdown("### Systemanalyse (kompakt)")

# 4 Spalten nebeneinander
col1, col2, col3, col4 = st.columns(4)

# Sprungantwort
with col1:
    st.markdown("**Sprungantwort**")
    t, y = step_response(closed_loop)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(t, y)
    ax.set_title("Sprung", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    st.pyplot(fig)

# Bode-Diagramm
with col2:
    st.markdown("**Bode-Diagramm**")
    mag, phase, omega = bode_plot(open_loop, w, plot=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 2), tight_layout=True)
    ax1.semilogx(omega, 20 * np.log10(mag))
    ax1.set_ylabel("dB", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.grid(True)
    ax2.semilogx(omega, np.degrees(phase))
    ax2.set_ylabel("°", fontsize=7)
    ax2.set_xlabel("ω", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.grid(True)
    st.pyplot(fig)

# Nyquist
with col3:
    st.markdown("**Nyquist-Diagramm**")
    fig, ax = plt.subplots(figsize=(3, 2))
    nyquist_plot(open_loop, omega=w, ax=ax)
    ax.set_title("Nyquist", fontsize=8)
    ax.tick_params(labelsize=6)
    st.pyplot(fig)

# Wurzelortskurve
with col4:
    st.markdown("**Wurzelortskurve**")
    fig, ax = plt.subplots(figsize=(3, 2))
    root_locus(open_loop, plot=True, ax=ax)
    ax.set_title("WOK", fontsize=8)
    ax.tick_params(labelsize=6)

    # Lead/Lag-Nullstelle und -Pol markieren
    if lead_lag_enable:
        lead_zero = -1 / z
        lead_pole = -1 / p
        ax.plot(np.real(lead_zero), np.imag(lead_zero), 'bo', label='Lead-Nullstelle')
        ax.plot(np.real(lead_pole), np.imag(lead_pole), 'rx', label='Lead-Pol')
        ax.legend(fontsize=6)

    st.pyplot(fig)


# ---- Stabilitätsinfo ---- #
st.markdown(f"### Stabilität: {'Stabil' if is_stable else 'Instabil'}")
st.markdown(f"### 🔁 Schwingfähigkeit: {'Schwingfähig' if is_oscillatory else 'Nicht schwingfähig'}")