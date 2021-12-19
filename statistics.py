import pickle
import matplotlib.pyplot as plt

with open("models_data/99_results.pkl", 'rb') as f:
    stats = pickle.load(f)

stats = stats[::942]
g = [row[2] for row in stats]
d = [row[3] for row in stats]
s = [row[6]/30 for row in stats]
o = [row[5] for row in stats]

print(o)

plt.plot(g, label="Gen")
plt.plot(d, label="Disc")
plt.plot(s, label="score")

plt.show()

