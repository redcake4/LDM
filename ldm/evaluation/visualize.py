from pathlib import Path
import numpy as np


def save_panel(path, source, target, prediction, segmentation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    union = (segmentation > 0.5).any(0)
    center = []
    for i in range(3):
        other_axes = tuple(j for j in range(3) if j != i)
        center.append(int(np.argmax(union.sum(axis=other_axes))) if union.any() else source.shape[i] // 2)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5), squeeze=False)
    for axis in range(3):
        for column, volume in enumerate((source, target, prediction, np.abs(prediction-target))):
            image = np.take(volume, center[axis], axis=axis)
            axes[axis, column].imshow(np.rot90(image), cmap="inferno" if column == 3 else "gray",
                                     vmin=0, vmax=0.4 if column == 3 else 1)
            axes[axis, column].axis("off")
            if axis == 0:
                axes[axis, column].set_title(("Source", "Target", "LDM", "Absolute error")[column])
    fig.tight_layout(pad=0.4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
