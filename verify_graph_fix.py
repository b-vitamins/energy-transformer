#!/usr/bin/env python3
"""Verify graph execution is fixed."""

import torch
import torch.nn as nn

from energy_transformer.spec import graph, realise
from energy_transformer.spec.combinators import Graph


class AddModule(nn.Module):
    """Simple module that adds a constant."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        """Add stored value to input tensor."""
        print(f"AddModule({self.value}) input shape: {x.shape}")
        return x + self.value


g = graph()
g = g.add_node("layer1", type("Layer1", (AddModule,), {})(1))
g = g.add_node("layer2", type("Layer2", (AddModule,), {})(10))
g = g.add_node("layer3", type("Layer3", (AddModule,), {})(100))

g = g.add_edge("input", "layer1")
g = g.add_edge("layer1", "layer2")
g = g.add_edge("layer2", "layer3")

graph_spec = Graph(nodes=g.nodes, edges=g.edges, inputs=["input"], outputs=["layer3"])

x = torch.tensor([0.0])
module = realise(graph_spec)
result = module(x)

print(f"\nInput: {x}")
print(f"Output: {result}")
print(f"Expected: {0 + 1 + 10 + 100} = 111")
print(f"Correct: {result.item() == 111}")
