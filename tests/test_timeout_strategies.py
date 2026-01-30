"""Tests for AdaptiveTimeout."""

from ai_fallback import AdaptiveTimeout


def test_default_timeout():
    t = AdaptiveTimeout(base_timeout=10.0)
    assert t.get_timeout() == 10.0


def test_backoff_on_retries():
    t = AdaptiveTimeout(base_timeout=10.0, backoff_factor=2.0)
    t1 = t.get_timeout(attempt=1)
    t2 = t.get_timeout(attempt=2)
    t3 = t.get_timeout(attempt=3)
    assert t2 == t1 * 2
    assert t3 == t1 * 4


def test_prompt_length_bonus():
    t = AdaptiveTimeout(base_timeout=10.0, prompt_length_factor=0.1)
    short = t.get_timeout(prompt_length=100)
    long = t.get_timeout(prompt_length=1000)
    assert long > short


def test_max_timeout_capped():
    t = AdaptiveTimeout(base_timeout=10.0, max_timeout=15.0, backoff_factor=3.0)
    assert t.get_timeout(attempt=5) == 15.0


def test_min_timeout_floor():
    t = AdaptiveTimeout(base_timeout=1.0, min_timeout=5.0)
    assert t.get_timeout() == 5.0


def test_adaptive_base_from_latencies():
    t = AdaptiveTimeout(base_timeout=5.0)
    for _ in range(10):
        t.record_latency(2.0)
    # p95 of [2.0]*10 = 2.0, adaptive base = 2.0*1.5=3.0, but base_timeout=5.0 wins
    assert t.get_timeout() == 5.0
    # Now with higher latencies
    t2 = AdaptiveTimeout(base_timeout=5.0)
    for _ in range(10):
        t2.record_latency(10.0)
    # p95=10.0, adaptive=15.0 > base 5.0
    assert t2.get_timeout() == 15.0


def test_reset():
    t = AdaptiveTimeout()
    t.record_latency(100.0)
    t.reset()
    assert t._recent_latencies == []
