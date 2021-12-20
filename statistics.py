import pickle
import matplotlib.pyplot as plt

with open("models_data/99_results.pkl", 'rb') as f:
    stats = pickle.load(f)
max_score = max([row[6] for row in stats])
print(max_score)
stats = stats[500:]
g = [row[2] for row in stats]
d = [row[3] for row in stats]
s = [row[6]/30 for row in stats]
o = [row[5] for row in stats]

print([s[5] for s in stats if s[6]>=max_score-2])

plt.plot(g, label="Gen")
plt.plot(d, label="Disc")
plt.plot(s, label="score")
plt.legend()
plt.xlim([30900, 41000])
plt.show()

