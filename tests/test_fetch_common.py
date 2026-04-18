from src.io.fetch_common import fetch_range_dates, load_fetch_config, raw_out_dir


def test_load_fetch_config():
    cfg = load_fetch_config()
    assert "fetch" in cfg
    assert cfg["fetch"]["date_range"]["start"] == "2010-01-01"
    assert cfg["fetch"]["date_range"]["end"] == "2025-12-31"


def test_fetch_range_dates():
    cfg = load_fetch_config()
    start, end = fetch_range_dates(cfg)
    assert start == "2010-01-01"
    assert end == "2025-12-31"
    assert len(end) == 10  # YYYY-MM-DD


def test_raw_out_dir():
    cfg = load_fetch_config()
    p = raw_out_dir(cfg, "noaa_tides")
    assert p.parts[-2:] == ("raw", "noaa_tides")
