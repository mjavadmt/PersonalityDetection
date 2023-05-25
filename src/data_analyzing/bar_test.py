import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from matplotlib.pyplot import figure

figure(figsize=(8, 8), dpi=80)

plt.bar(range(5), [5, 2, 3, 4, 3], tick_label=[get_display(reshape(label)) for label in ["کیری", "ممدد", "کمترررر", "سلام", "ممد"]])
plt.title(f"most frequent words in tweets", fontweight="bold")
plt.xticks(rotation=90)
plt.show()
