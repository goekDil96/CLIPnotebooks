import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
plt.rc('font', family='serif',size=16)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=16)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


# Erzeugen eines Arrays von Werten im Intervall [0, 1] (ohne 0, da ln(0) undefiniert ist)
x = np.linspace(0.0001, 1, 1000)
y = -np.log(x)

# Plotten der Funktion
plt.figure(figsize=(12, 6))
plt.plot(x, y, label=r'$- \ln(x)$')
plt.xlabel('x')
plt.ylabel(r'$- \ln(x)$')
plt.ylim(bottom=0)  # Start der y-Achse bei 0 für bessere Darstellung
plt.legend()
plt.grid(True)


# Save the figure
plt.savefig(f"target_images/minus_log.png", bbox_inches='tight')
