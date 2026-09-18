from drift import population_stability_index


def test_population_stability_index_is_zero_for_identical_distributions():
    assert population_stability_index({"Supplies": 50, "Meals": 50}, {"Supplies": 50, "Meals": 50}) == 0.0


def test_population_stability_index_detects_distribution_shift():
    value = population_stability_index({"Supplies": 90, "Meals": 10}, {"Supplies": 10, "Meals": 90})
    assert value > 1.0
