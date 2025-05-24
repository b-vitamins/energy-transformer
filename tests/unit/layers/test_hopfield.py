import math

from energy_transformer.layers.hopfield import he_scaled_init_std, get_energy_transformer_init_std


def test_he_scaled_init_std_formula():
    assert math.isclose(he_scaled_init_std(4), math.sqrt(2.0 / 4))
    assert math.isclose(he_scaled_init_std(4, 8), math.sqrt(2.0 / 8))
    assert math.isclose(get_energy_transformer_init_std(3, 6), he_scaled_init_std(3, 6))
