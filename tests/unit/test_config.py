from pathlib import Path

from agmlib.config import load_config


def test_load_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("{}", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.replay.batch_size >= 1
    assert 0.0 <= cfg.replay.alpha <= 1.0
    assert cfg.kernel_smoothing.k >= 1
    assert cfg.kernel_smoothing.sigma_min <= cfg.kernel_smoothing.sigma_max

