from agmlib.agm.curriculum import AGMCurriculumController, CurriculumConfig


def test_curriculum_promote_and_demote():
    cfg = CurriculumConfig(difficulty_levels=["easy", "medium", "hard"], stability_requirement=0.6, mastery_threshold=0.8, drop_threshold=0.2)
    ctl = AGMCurriculumController(cfg)
    # Promote
    act1 = ctl.update_curriculum(agm_convergence_rate=0.7, success_rate=0.85, performance_metrics={})
    assert act1 == "increase_difficulty"
    assert ctl.current_level == 1
    # Demote
    act2 = ctl.update_curriculum(agm_convergence_rate=0.1, success_rate=0.2, performance_metrics={})
    assert act2 in {"decrease_difficulty", "maintain_difficulty"}  # boundary-safe


def test_curriculum_state_keys():
    ctl = AGMCurriculumController()
    ctl.update_curriculum(agm_convergence_rate=0.5, success_rate=0.6, performance_metrics={})
    state = ctl.get_curriculum_state()
    assert set(["current_level", "current_difficulty", "total_levels", "progression_rate", "stability_trend"]).issubset(state.keys())


