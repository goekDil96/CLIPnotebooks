import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
plt.rc('font', family='serif',size=24)
matplotlib.rc('text', usetex=True)
matplotlib.rc('legend', fontsize=24)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

def softmax(z, T=1.0):
    exp_z = np.exp(z / T)
    return exp_z / np.sum(exp_z)

# Eingabevektor
z = np.array([1.0, 2.0, 3.0, 4.0, 1.0])

# Temperaturwerte
temperatures = [0.01, 0.5, 1.0, 2.0, 5.0, 6.0]

# Plot erstellen
fig, axs = plt.subplots(2, 3, figsize=(25, 17))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Plot the softmax probabilities for each temperature
for i, T in enumerate(temperatures):
    probabilities = softmax(z, T)
    axs[i].bar(range(len(z)), probabilities)
    axs[i].set_title(f'Softmax with Temperature tau = {T:.2f}')
    axs[i].set_ylim(0, 1)
    axs[i].grid(True)

# Remove the extra subplots
for i in range(len(temperatures), len(axs)):
    fig.delaxes(axs[i])

plt.xlabel('Index')
plt.ylabel('Probability')
plt.tight_layout()

# Save the figure
plt.savefig("target_images/temp_softmax.png", bbox_inches='tight')
plt.show()
