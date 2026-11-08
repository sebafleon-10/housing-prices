"""Portfolio fan-card cover: real sale prices vs the placeholder spike."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = json.load(open(sys.argv[1]))["hist"]
real, fake = np.array(h["real"]), np.array(h["placeholder"])
W, H = 1536, 928
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); fig.patch.set_facecolor("#0b0b0d")
ax = fig.add_axes([0.06, 0.38, 0.88, 0.52]); ax.set_facecolor("#0b0b0d"); ax.axis("off")
x = np.arange(len(real)); m = max(real.max(), fake.max())
ax.bar(x - 0.2, real / m, width=0.38, color="white", alpha=0.45, linewidth=0)
ax.bar(x + 0.2, fake / m, width=0.38, color="#9D9FFF", linewidth=0)
ax.set_ylim(0, 1.05); ax.set_xlim(-0.8, len(real) - 0.2)
fig.savefig(sys.argv[2], facecolor=fig.get_facecolor())
