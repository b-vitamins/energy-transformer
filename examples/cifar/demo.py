"""Demonstrate topology-aware simplex generation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import Polygon  # Import Polygon from correct module

from energy_transformer.layers.simplicial import (
    TopologyAwareSimplexGenerator,
)

EDGE_SIZE = 2
TRIANGLE_SIZE = 3

# Type alias for clarity
Coordinate = tuple[int, int]
Simplex = list[int]


def visualize_topology(grid_size: int = 8) -> None:
    """Visualize topology-aware vs random simplices."""
    # Generate coordinates
    coords: list[Coordinate] = [
        (i, j) for i in range(grid_size) for j in range(grid_size)
    ]
    num_patches = len(coords)

    # Topology-aware simplices
    generator = TopologyAwareSimplexGenerator(
        coordinates=coords,
        k_neighbors=6,
        include_delaunay=True,
    )
    topo_simplices = generator.generate(
        num_vertices=num_patches,
        max_dim=2,
        budget=0.15,  # budget_fraction
    )

    np.random.seed(42)
    n_random = len(topo_simplices)
    random_simplices: list[Simplex] = []

    for _ in range(n_random // 2):
        # Random edge
        i, j = np.random.choice(num_patches, 2, replace=False)
        random_simplices.append(sorted([int(i), int(j)]))

    for _ in range(n_random // 4):
        # Random triangle
        i, j, k = np.random.choice(num_patches, 3, replace=False)
        random_simplices.append(sorted([int(i), int(j), int(k)]))

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
        edges = [s for s in simplices if len(s) == EDGE_SIZE]
        for edge in edges:
            p1, p2 = coords[edge[0]], coords[edge[1]]
            ax.plot(
                [p1[1], p2[1]],
                [p1[0], p2[0]],
                "b-",
                alpha=0.4,
                linewidth=1,
            )

        # Draw triangles
        triangles = [s for s in simplices if len(s) == TRIANGLE_SIZE]
        for tri in triangles:
            points = [coords[idx] for idx in tri]
            triangle = Polygon(
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
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
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
        edges = [s for s in simplices if len(s) == EDGE_SIZE]
        distances: list[float] = []

        for edge in edges:
            coord1, coord2 = coords[edge[0]], coords[edge[1]]
            p1_array = np.array(coord1)
            p2_array = np.array(coord2)
            distances.append(float(np.linalg.norm(p1_array - p2_array)))

        if distances:
            print(f"{name}:")
            print(f"  Average distance: {np.mean(distances):.2f}")
            print(f"  Max distance: {np.max(distances):.2f}")
            print()


def main() -> None:
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
