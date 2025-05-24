import torch

from energy_transformer.spec import PosEmbedSpec
from energy_transformer.spec.realise import SpecInfo, realise


def test_positional_embedding_init_std() -> None:
    torch.manual_seed(0)
    spec = PosEmbedSpec(include_cls=False, init_std=0.1)
    info = SpecInfo(embedding_dim=8, token_count=4)
    module = realise(spec, info)
    assert module is not None
    # Standard deviation should be close to the specified value
    assert torch.isclose(module.pos_embed.std(), torch.tensor(0.1), atol=0.02)
