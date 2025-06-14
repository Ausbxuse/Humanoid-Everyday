import matplotlib.pyplot as plt

data = {
    "Basic Manipulation": {
        "Pick-and-Place": 35,
        "Pouring & Transferring": 10,
        "Pushing & Pulling": 8,
        "Pressing & Clicking": 6,
        "Rotating & Tilting": 5,
        "Stacking & Aligning": 4,
        "Opening & Closing": 3,
    },
    "Loco-Manipulation": {
        "Walk & Interact": 18,
        "Walk & PnP": 12,
        "Walk & Furniture": 6,
        "Walk & Clean": 5,
        "Walk & Transport": 4,
    },
    "H&R Interaction": {
        "Give to Human": 12,
        "Receive from Human": 12,
        "Social Gesture": 9,
        "Hold & Return": 8,
    },
    "Articulate Object": {
        "Flipping/Hinge": 8,
        "Button Pressing": 7,
        "Drawer & Tray": 6,
        "Lid & Cover": 6,
        "Compression": 6,
        "Rotation/Tilting": 5,
        "Handle Turning": 3,
        "Door Closing": 3,
        "Extension/Collapse": 2,
        "Tool Removal": 2,
    },
    "Tool Use": {
        "Cleaning Tools": 7,
        "Impact Tools": 5,
        "Duster/Eraser": 4,
        "Dispensing": 4,
        "Removal/Insertion": 3,
    },
    "Deformable Object": {
        "Fold/Unfold": 6,
        "Squeeze/Compress": 6,
        "Stretch/Flip": 5,
        "Smooth/Adjust": 5,
    },
    "High Precision": {
        "Precise Placement": 3,
        "Insertion/Removal": 3,
        "Micro Button": 2,
        "Fine Stacking": 1,
    },
}

main_colors = {
    "Basic Manipulation": "#50bad8",
    "Loco-Manipulation": "#2c98f8",
    "H&R Interaction": "#d94064",
    "Articulate Object": "#7d59bd",
    "Tool Use": "#d77a42",
    "Deformable Object": "#d9a439",
    "High Precision": "#79af57",
}

x = []
heights = []
bar_colors = []
labels = []
regions = []
current_x = 0

for cat, subcats in data.items():
    sorted_sub = sorted(subcats.items(), key=lambda kv: kv[1], reverse=True)
    start = current_x
    for name, count in sorted_sub:
        x.append(current_x)
        heights.append(count)
        bar_colors.append(main_colors[cat])
        labels.append(name)
        current_x += 1
    end = current_x - 1
    regions.append((start - 0.5, end + 0.5, cat))
    current_x += 0.5

fig, ax = plt.subplots(figsize=(14, 6))

for start, end, cat in regions:
    ax.axvspan(start, end, color=main_colors[cat], alpha=0.1)

ax.bar(x, heights, color=bar_colors, edgecolor="black")

ax.set_yscale("log")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_ylabel("# Tasks")
ax.set_title("Humanoid Everyday Dataset Distribution")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", which="both", linestyle="--", linewidth=0.5, alpha=0.7)

ax.set_ylim(1, max(heights) * 1.5)

y_top = ax.get_ylim()[1]
for start, end, cat in regions:
    edge = end - 1
    ax.text(
        edge,
        y_top * 0.92,
        cat,
        ha="center",
        va="top",
        fontsize=13,
        color=main_colors[cat],
        rotation=90,
        alpha=0.7,
    )

plt.tight_layout()
plt.savefig("data_dist.png", dpi=300, bbox_inches="tight")
