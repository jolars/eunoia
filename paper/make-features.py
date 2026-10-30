"""Generate the paper's annotation figure with eunoia-py and Matplotlib.

Run with a Python environment containing eunoia-py, for example:
    ../eunoia-py/.devenv/state/venv/bin/python paper/make-features.py

The output defaults to paper/images/ regardless of the working directory.
The two narrow pairwise regions use Matplotlib callouts to keep names legible.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import eunoia as eu
import matplotlib.pyplot as plt

# Match the standalone plot width and label size in make-figures.mjs.
PLOT_MM = 70
LABEL_MM = 3
MEMBER_MM = 2.5
MM_PER_INCH = 25.4
POINTS_PER_MM = 72 / MM_PER_INCH
COLORS = {"A": "#ffffff", "B": "#d9d9d9", "C": "#add8e6"}

REGION_MEMBERS = {
    "A": [f"gene{i:02}" for i in range(1, 5)],
    "B": [f"gene{i:02}" for i in range(5, 8)],
    "C": [f"gene{i:02}" for i in range(8, 12)],
    "A&B": ["gene12", "gene13"],
    "A&C": ["gene14"],
    "B&C": ["gene15"],
    "A&B&C": [f"gene{i:02}" for i in range(16, 22)],
}
CALLOUTS = {"A&C": (0.02, 0.85, "left"), "B&C": (0.98, 0.85, "right")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
    )
    args = parser.parse_args()
    memberships = {
        name: [
            member
            for region, members in REGION_MEMBERS.items()
            if name in region.split("&")
            for member in members
        ]
        for name in COLORS
    }
    diagram = eu.euler(memberships, shape="ellipse", seed=1, n_threads=1)

    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "eunoia-paper-features",
            "pdf.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(2 * PLOT_MM / MM_PER_INCH, (PLOT_MM + 6) / MM_PER_INCH),
        )
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9, wspace=0.05)
        style = {
            "colors": COLORS,
            "fills": {region: {"alpha": 1} for region in REGION_MEMBERS},
            "edges": {"linewidth": 0.5},
            "labels": {
                "fontsize": LABEL_MM * POINTS_PER_MM,
                "set_position": "outside",
            },
        }
        # Missing names or units would make the two panels disagree.
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            diagram.plot(
                ax=axes[0],
                **style,
                glyphs={"obstacle_policy": "strict", "linewidth": 0.4},
            )
            # These narrow overlaps cannot hold their names at a readable size.
            # Callouts keep them from shrinking every packed label.
            packed = replace(
                diagram,
                members={
                    region: names
                    for region, names in REGION_MEMBERS.items()
                    if region not in CALLOUTS
                },
            )
            packed.plot(
                ax=axes[1],
                **style,
                members={
                    "mode": "packed",
                    "fontsize": MEMBER_MM * POINTS_PER_MM,
                    "min_scale": 0.85,
                    "gap": 0.4,
                    "target_aspect_ratio": 1,
                    "obstacle_policy": "strict",
                },
            )
        for region, (x, y, align) in CALLOUTS.items():
            axes[1].annotate(
                "\n".join(REGION_MEMBERS[region]),
                xy=diagram.plot_data["region_anchors"][region],
                xytext=(x, y),
                textcoords="axes fraction",
                ha=align,
                va="center",
                fontsize=MEMBER_MM * POINTS_PER_MM,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "black"},
            )
        for ax, title in zip(axes, ("Glyphs", "Member labels"), strict=True):
            ax.set_title(title, fontsize=LABEL_MM * POINTS_PER_MM, fontweight="bold")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Preserve the canvas dimensions so LaTeX keeps the physical font sizes.
        for extension, metadata in (
            ("svg", {"Date": None}),
            ("pdf", {"CreationDate": None, "ModDate": None}),
        ):
            path = args.output_dir / f"features.{extension}"
            fig.savefig(path, metadata=metadata)
            if extension == "svg":
                # Keep regenerated paths clean under git diff --check.
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
            print(f"wrote {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
