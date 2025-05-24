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
    z = st.sidebar.slider("z (Lead-Nullstelle, 1/z = Position)", -10.0, 10.0, 1.0)
    p = st.sidebar.slider("p (Lead-Pol, 1/p = Position)", -10.0, 10.0, 2.0)
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
# Bode-Diagramm mit erweiterten Features
with col2:
    st.markdown("**Bode-Diagramm**")
    mag, phase, omega = bode_plot(open_loop, w, plot=False)
    phase_deg = np.degrees(phase)
    gain_db = 20 * np.log10(mag)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 2), tight_layout=True)

    # Amplitudenplot
    ax1.semilogx(omega, gain_db)
    ax1.set_ylabel("dB", fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.grid(True)

    # Phasenplot
    ax2.semilogx(omega, phase_deg)
    ax2.set_ylabel("°", fontsize=7)
    ax2.set_xlabel("ω", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.grid(True)

    padding = 10
    # Grenzen des Phasenbereichs (auf nächste 45° runden)
    def round_down_45(x):
        return 45 * np.floor((x - padding) / 45)
    def round_up_45(x):
        return 45 * np.ceil((x + padding) / 45)

    phase_min = np.nanmin(phase_deg)
    phase_max = np.nanmax(phase_deg)
    ymin = round_down_45(phase_min)
    ymax = round_up_45(phase_max)
    ax2.set_ylim(ymin, ymax)

    # horizontale Linien alle 45°
    yticks = np.arange(ymin, ymax + 1, 45)
    ax2.set_yticks(yticks)
    for y in yticks:
        ax2.axhline(y, color='gray', linestyle='--', linewidth=0.4)

    # Schnittpunkte mit 45°-Linien markieren
    for target_phase in yticks:
        for i in range(len(phase_deg) - 1):
            p1, p2 = phase_deg[i], phase_deg[i + 1]
            if (p1 - target_phase) * (p2 - target_phase) < 0:
                w1, w2 = omega[i], omega[i + 1]
                alpha = (target_phase - p1) / (p2 - p1)
                w_cross = w1 + alpha * (w2 - w1)
                ax2.hlines(target_phase, w_cross * 0.98, w_cross * 1.02, colors='black', linewidth=1)

    # Quartalsbeschriftung – nur einmal pro Quartal
    quartal_labels = {
        "Q1": 45,
        "Q2": 135,
        "Q3": -135,
        "Q4": -45
    }
    x_middle = omega[len(omega) // 2]
    for q, phi in quartal_labels.items():
        if ymin <= phi <= ymax:
            ax2.text(x_middle, phi, q, fontsize=6, color='darkred', ha='center', va='center', rotation=0)


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

    # Wurzelschwerpunkt berechnen und einzeichnen
    # Nullstellen und Pole des offenen Regelkreises (open_loop)
    zeros_ol = open_loop.zeros()
    poles_ol = open_loop.poles()

    # Anzahl Pole und Nullstellen
    n_poles = len(poles_ol)
    n_zeros = len(zeros_ol)

    # Falls n_poles != n_zeros (sonst Division durch 0 vermeiden)
    if n_poles != n_zeros:
        sum_poles = np.sum(poles_ol)
        sum_zeros = np.sum(zeros_ol)
        ws = (sum_poles - sum_zeros) / (n_poles - n_zeros)
        ax.plot(np.real(ws), np.imag(ws), 'md', markersize=10, label='Wurzelschwerpunkt')

    ax.legend(fontsize=6)

    st.pyplot(fig)



# ---- Stabilitätsinfo ---- #
st.markdown(f"### Stabilität: {'Stabil' if is_stable else 'Instabil'}")
st.markdown(f"### 🔁 Schwingfähigkeit: {'Schwingfähig' if is_oscillatory else 'Nicht schwingfähig'}")