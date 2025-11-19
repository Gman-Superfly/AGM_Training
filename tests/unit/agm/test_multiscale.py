from agmlib.agm.multiscale import MultiScaleAGMFramework


def test_multiscale_hierarchical_keys_present():
    ms = MultiScaleAGMFramework()
    ms.update_all_trackers({"convergence_rate": 0.5, "mean_spread": 0.1})
    out = ms.compute_hierarchical_adaptation()
    assert set(out.keys()) == {"immediate", "tactical", "strategic"}


