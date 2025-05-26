#!/usr/bin/env python3
"""Demonstrate topology-aware simplex generation."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from energy_transformer.layers.simplicial import _topology_aware_simps


def visualize_topology(grid_size=8):
    """Visualize topology-aware vs random simplices."""
    # Generate coordinates
    coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    num_patches = len(coords)

    # Topology-aware simplices
    topo_simplices = _topology_aware_simps(
        coordinates=coords,
        k_neighbors=6,
        include_delaunay=True,
        max_dim=2,
        budget_fraction=0.15,
    )

    # Random simplices (simplified)
    np.random.seed(42)
    n_random = len(topo_simplices)
    random_simplices = []

    for _ in range(n_random // 2):
        # Random edge
        i, j = np.random.choice(num_patches, 2, replace=False)
        random_simplices.append(sorted([i, j]))

    for _ in range(n_random // 4):
        # Random triangle
        i, j, k = np.random.choice(num_patches, 3, replace=False)
        random_simplices.append(sorted([i, j, k]))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, simplices, title in [
        (ax1, topo_simplices, "Topology-Aware (k-NN + Delaunay)"),
        (ax2, random_simplices, "Random Simplices"),
    ]:
        # Draw patches
        for idx, (i, j) in enumerate(coords):
            rect = patches.Rectangle(
                (j - 0.4, i - 0.4),
                0.8,
                0.8,
                linewidth=1,
                edgecolor="black",
                facecolor="lightblue",
                alpha=0.5,
            )
            ax.add_patch(rect)
            ax.text(j, i, str(idx), ha="center", va="center", fontsize=6)

        # Draw edges
        edges = [s for s in simplices if len(s) == 2]
        for edge in edges:
            p1, p2 = coords[edge[0]], coords[edge[1]]
            ax.plot(
                [p1[1], p2[1]], [p1[0], p2[0]], "b-", alpha=0.4, linewidth=1
            )

        # Draw triangles
        triangles = [s for s in simplices if len(s) == 3]
        for tri in triangles:
            points = [coords[idx] for idx in tri]
            triangle = plt.Polygon(
                [(p[1], p[0]) for p in points],
                facecolor="red",
                alpha=0.15,
                edgecolor="red",
                linewidth=0.5,
            )
            ax.add_patch(triangle)

        ax.set_xlim(-1, grid_size)
        ax.set_ylim(-1, grid_size)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.text(
            0.02,
            0.98,
            f"Edges: {len(edges)}\nTriangles: {len(triangles)}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig("topology_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Distance analysis
    print("Edge Distance Analysis:")
    print("-" * 40)

    for name, simplices in [
        ("Topology-aware", topo_simplices),
        ("Random", random_simplices),
    ]:
        edges = [s for s in simplices if len(s) == 2]
        distances = []

        for edge in edges:
            p1, p2 = np.array(coords[edge[0]]), np.array(coords[edge[1]])
            distances.append(np.linalg.norm(p1 - p2))

        if distances:
            print(f"{name}:")
            print(f"  Average distance: {np.mean(distances):.2f}")
            print(f"  Max distance: {np.max(distances):.2f}")
            print()


def main():
    """Run demonstration."""
    print("Topology-Aware Simplex Generation Demo")
    print("=" * 40)

    visualize_topology(grid_size=8)

    print("\nKey Insights:")
    print("- Topology-aware connects nearby patches (local patterns)")
    print("- Random connects arbitrary patches (no spatial structure)")
    print("- Triangles in topology capture actual image corners")
    print("- This spatial bias helps vision tasks significantly")


if __name__ == "__main__":
    main()
