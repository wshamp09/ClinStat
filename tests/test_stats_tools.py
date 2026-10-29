"""Validate the calculators against published values and Monte Carlo simulation.

Each simulation generates data under the stated design, applies the actual test
the method describes, and checks the empirical rejection rate against the
analytic power (tolerance ~3.5 Monte Carlo SEs).
"""
import math

import numpy as np
import pytest
from scipy import stats

from clinstat.stats_tools import (
    adjust_pvalues,
    binary_sample_size,
    continuous_sample_size,
    correlated_eyes_adjustment,
    inflate_for_dropout,
)

SIMS = 20_000
RNG = np.random.default_rng(20261025)


def _tol(p: float, sims: int = SIMS) -> float:
    return 3.5 * math.sqrt(p * (1 - p) / sims) + 0.003


# ---------------------------------------------------------------- reference values
def test_parallel_superiority_reference():
    # Classic: d = 0.5 SD, alpha 0.05 two-sided, 80% power -> 64 per group (t-test)
    r = continuous_sample_size("parallel", "superiority", delta=0.5, sd=1.0, power=0.8)
    assert r.outputs["n_control"] == 64 and r.outputs["n_treatment"] == 64


def test_crossover_tost_matches_powertost():
    # PowerTOST::sampleN.TOST(CV = 0.3, theta0 = 0.95) -> n = 40, power 0.8158
    r = continuous_sample_size("crossover_2x2", "equivalence", scale="ratio", gmr=0.95, cv=0.30, power=0.8)
    assert r.outputs["n_total"] == 40
    assert abs(r.outputs["power"] - 0.8158) < 0.001


def test_binary_superiority_reference():
    # Fleiss: p1 = 0.6, p2 = 0.8, alpha 0.05 two-sided, 80% -> 82 per group; with CC -> 91
    assert binary_sample_size(0.6, 0.8).outputs["n_control"] == 82
    assert binary_sample_size(0.6, 0.8, continuity_correction=True).outputs["n_control"] == 91


def test_solve_n_then_power_roundtrip():
    r = continuous_sample_size("paired", "superiority", delta=2.0, sd=5.0, power=0.9)
    n = r.outputs["n_pairs"]
    assert continuous_sample_size("paired", "superiority", delta=2.0, sd=5.0, n=n).outputs["power"] >= 0.9
    assert continuous_sample_size("paired", "superiority", delta=2.0, sd=5.0, n=n - 1).outputs["power"] < 0.9


def test_crossover_n_is_even():
    for d in (0.3, 0.45, 0.61):
        assert continuous_sample_size("crossover_2x2", "superiority", delta=d, sd=1.0).outputs["n_total"] % 2 == 0


# ---------------------------------------------------------------- simulations
def test_sim_parallel_superiority():
    n, delta, sd = 30, 0.6, 1.0
    analytic = continuous_sample_size("parallel", "superiority", delta=delta, sd=sd, n=n).outputs["power"]
    x = RNG.normal(0, sd, (SIMS, n))
    y = RNG.normal(delta, sd, (SIMS, n))
    p = stats.ttest_ind(y, x, axis=1).pvalue
    assert abs((p < 0.05).mean() - analytic) < _tol(analytic)


def test_sim_parallel_unequal_allocation():
    n_c, ratio, delta, sd = 20, 2.0, 0.6, 1.0
    analytic = continuous_sample_size("parallel", "superiority", delta=delta, sd=sd, n=n_c, ratio=ratio).outputs["power"]
    x = RNG.normal(0, sd, (SIMS, n_c))
    y = RNG.normal(delta, sd, (SIMS, int(n_c * ratio)))
    p = stats.ttest_ind(y, x, axis=1).pvalue
    assert abs((p < 0.05).mean() - analytic) < _tol(analytic)


def test_sim_crossover_superiority():
    # AB/BA: analyse period differences d_i = (y1 - y2)/2 between sequences.
    N, delta, sd_w, sd_b = 24, 0.5, 1.0, 2.0
    analytic = continuous_sample_size("crossover_2x2", "superiority", delta=delta, sd=sd_w, n=N).outputs["power"]
    half = N // 2
    subj = RNG.normal(0, sd_b, (SIMS, N))
    period = 0.3
    e1, e2 = RNG.normal(0, sd_w, (SIMS, N)), RNG.normal(0, sd_w, (SIMS, N))
    trt_p1 = np.r_[np.ones(half), np.zeros(half)]  # sequence AB gets A in period 1
    y1 = subj + delta * trt_p1 + e1
    y2 = subj + period + delta * (1 - trt_p1) + e2
    dd = (y1 - y2) / 2
    p = stats.ttest_ind(dd[:, :half], dd[:, half:], axis=1).pvalue
    assert abs((p < 0.05).mean() - analytic) < _tol(analytic)


def test_sim_paired_noninferiority():
    n, delta, sd, margin = 40, 0.0, 4.0, 2.0
    analytic = continuous_sample_size("paired", "noninferiority", delta=delta, sd=sd, n=n, margin=margin).outputs["power"]
    d = RNG.normal(delta, sd, (SIMS, n))
    t = (d.mean(1) + margin) / (d.std(1, ddof=1) / math.sqrt(n))
    emp = (t > stats.t.ppf(0.975, n - 1)).mean()
    assert abs(emp - analytic) < _tol(analytic)


def test_sim_parallel_equivalence():
    n, delta, sd, margin = 50, 0.5, 3.0, 2.0
    analytic = continuous_sample_size("parallel", "equivalence", delta=delta, sd=sd, n=n, margin=margin).outputs["power"]
    x = RNG.normal(0, sd, (SIMS, n))
    y = RNG.normal(delta, sd, (SIMS, n))
    diff = y.mean(1) - x.mean(1)
    sp = np.sqrt((x.var(1, ddof=1) + y.var(1, ddof=1)) / 2)
    se = sp * math.sqrt(2 / n)
    tc = stats.t.ppf(0.95, 2 * n - 2)
    emp = (((diff + margin) / se > tc) & ((diff - margin) / se < -tc)).mean()
    assert abs(emp - analytic) < _tol(analytic) + 0.005


def test_sim_binary_superiority():
    n, pc, pt = 120, 0.55, 0.72
    analytic = binary_sample_size(pc, pt, n_control=n).outputs["power"]
    xc = RNG.binomial(n, pc, SIMS) / n
    xt = RNG.binomial(n, pt, SIMS) / n
    pbar = (xc + xt) / 2
    z = (xt - xc) / np.sqrt(pbar * (1 - pbar) * 2 / n)
    emp = (np.abs(z) > stats.norm.ppf(0.975)).mean()
    assert abs(emp - analytic) < _tol(analytic) + 0.01


def test_sim_binary_noninferiority():
    n, pc, pt, m = 200, 0.85, 0.85, 0.10
    analytic = binary_sample_size(pc, pt, "noninferiority", margin=m, n_control=n).outputs["power"]
    xc = RNG.binomial(n, pc, SIMS) / n
    xt = RNG.binomial(n, pt, SIMS) / n
    se = np.sqrt(xc * (1 - xc) / n + xt * (1 - xt) / n)
    emp = ((xt - xc + m) / se > stats.norm.ppf(0.975)).mean()
    assert abs(emp - analytic) < _tol(analytic) + 0.01


# ---------------------------------------------------------------- other calculators
def test_multiplicity_hand_calculations():
    p = [0.01, 0.04, 0.03, 0.005]
    assert adjust_pvalues(p, "bonferroni").outputs["adjusted"] == [0.04, 0.16, 0.12, 0.02]
    assert adjust_pvalues(p, "holm").outputs["adjusted"] == [0.03, 0.06, 0.06, 0.02]
    assert adjust_pvalues(p, "hochberg").outputs["adjusted"] == [0.03, 0.04, 0.04, 0.02]
    assert adjust_pvalues(p, "bh").outputs["adjusted"] == [0.02, 0.04, 0.04, 0.02]
    fs = adjust_pvalues([0.01, 0.20, 0.001], "fixed_sequence")
    assert fs.outputs["reject"] == [True, False, False]


def test_correlated_eyes_and_dropout():
    r = correlated_eyes_adjustment(n_independent_eyes=100, eyes_per_subject=2, icc=0.6)
    assert r.outputs == {"design_effect": 1.6, "eyes_needed": 160, "subjects_needed": 80}
    assert inflate_for_dropout(64, 0.15, multiple_of=4).outputs["enrol_n"] == 76
    assert inflate_for_dropout(80, 0.20).outputs["enrol_n"] == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(design="parallel", hypothesis="noninferiority", delta=0, sd=1),  # no margin
        dict(design="parallel", hypothesis="equivalence", delta=3, sd=1, margin=2),  # delta outside margins
        dict(design="bogus", hypothesis="superiority", delta=1, sd=1),
        dict(design="parallel", hypothesis="superiority", delta=0, sd=1),
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        continuous_sample_size(**kwargs)
