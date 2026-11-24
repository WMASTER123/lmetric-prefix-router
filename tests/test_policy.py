import pytest

from lmetric_prefix_router.policy import ExternalBalancer, Replica, choose_rank


def test_protects_single_owner_even_under_pressure():
    assert choose_rank(4096, (Replica(2048, 16, 8000, 0), Replica(0, 0, 0, 0))) == 0


@pytest.mark.parametrize("second_hit", [None, 0])
def test_delegates_unknown_or_zero(second_hit):
    assert (
        choose_rank(128, (Replica(0, 4, 128, 0), Replica(second_hit, 0, 0, 0))) is None
    )


def test_freshness_boundary_and_stale_equal_hits():
    a = Replica(2048, 5, 0, 0.6)
    assert choose_rank(3000, (a, Replica(2048, 1, 0, 0.6))) is None
    assert choose_rank(3000, (a, Replica(2048, 1, 0, 0.61))) == 1


def test_more_than_two_ranks_and_fractional_delegated_load():
    ranks = (Replica(0, 0, 0, 0), Replica(128, 0.5, 90, 0), Replica(128, 1, 0, 0))
    assert choose_rank(256, ranks) == 2


def test_invalid_snapshot_rejected():
    with pytest.raises(ValueError):
        choose_rank(128, (Replica(2048, 0, 0, 0),))


def test_same_external_fallback_in_both_arms():
    baseline, combined = ExternalBalancer(), ExternalBalancer()
    ranks = (Replica(128, 2, 0, 0), Replica(128, 2, 0, 0))
    for expected in (0, 1, 0):
        assert baseline.choose(256, ranks, use_prefix=False) == expected
        assert combined.choose(256, ranks, use_prefix=True) == expected
    hot = (Replica(128, 9, 0, 0), Replica(0, 0, 0, 0))
    assert baseline.choose(256, hot, use_prefix=False) == 1
    assert combined.choose(256, hot, use_prefix=True) == 0
