"""Vetted statistical calculators.

The agent calls these instead of doing arithmetic in prose, and the
Calculators tab exposes the same functions directly. Each returns a
``CalcResult`` that records the inputs, outputs, method and assumptions so an
answer can cite exactly how a number was produced. ``tests/test_stats_tools.py``
checks every power formula against Monte Carlo simulation of the actual test.

Conventions
-----------
* ``delta`` is the true difference *treatment - control* on the analysis scale.
* Non-inferiority assumes higher values are better: H0 is delta <= -margin.
* Continuous designs use the t distribution (exact power via the noncentral t;
  equivalence via the noncentral-t approximation used by PowerTOST "nct").
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

MAX_N = 200_000


@dataclass
class CalcResult:
    calculator: str
    inputs: dict
    outputs: dict
    method: str
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        lines = [f"Calculator: {self.calculator}"]
        lines.append("Inputs: " + ", ".join(f"{k}={_fmt(v)}" for k, v in self.inputs.items() if v is not None))
        lines.append("Results: " + ", ".join(f"{k}={_fmt(v)}" for k, v in self.outputs.items()))
        lines.append(f"Method: {self.method}")
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _check_prob(name: str, value: float, low_open: bool = True) -> None:
    if not (0 < value < 1):
        raise ValueError(f"{name} must be strictly between 0 and 1 (got {value})")


# ==========================================================================
# Continuous endpoints: parallel, 2x2 crossover, paired
# ==========================================================================

_DESIGNS = {
    "parallel": "two-arm parallel group; sd = common between-subject SD",
    "crossover_2x2": "AB/BA crossover, equal sequence sizes; sd = within-subject (residual) SD",
    "paired": "paired / contralateral design; sd = SD of within-pair differences",
}


def _se_df(design: str, n: int, sd: float, ratio: float) -> tuple[float, int, dict]:
    """Standard error of the estimated difference and its df.

    ``n`` is the control-group size for parallel designs and the total number of
    subjects (or pairs) otherwise.
    """
    if design == "parallel":
        n_c = n
        n_t = math.ceil(ratio * n)
        return sd * math.sqrt(1 / n_c + 1 / n_t), n_c + n_t - 2, {"n_control": n_c, "n_treatment": n_t, "n_total": n_c + n_t}
    if design == "crossover_2x2":
        # Var(d_hat) = (sd_w^2 / 2)(1/n1 + 1/n2) = 2 sd_w^2 / N for n1 = n2 = N/2
        return sd * math.sqrt(2 / n), n - 2, {"n_total": n, "n_per_sequence": n // 2}
    if design == "paired":
        return sd / math.sqrt(n), n - 1, {"n_pairs": n}
    raise ValueError(f"design must be one of {list(_DESIGNS)}")


def _power_t(hypothesis: str, delta: float, se: float, df: int, alpha: float, sides: int,
             lower: float | None, upper: float | None) -> float:
    nct = stats.nct
    if hypothesis == "superiority":
        ncp = delta / se
        if sides == 2:
            t = stats.t.ppf(1 - alpha / 2, df)
            return float(1 - nct.cdf(t, df, ncp) + nct.cdf(-t, df, ncp))
        t = stats.t.ppf(1 - alpha, df)
        return float(1 - nct.cdf(t, df, abs(ncp)))
    if hypothesis == "noninferiority":
        t = stats.t.ppf(1 - alpha, df)
        return float(1 - nct.cdf(t, df, (delta - lower) / se))
    if hypothesis == "equivalence":
        t = stats.t.ppf(1 - alpha, df)
        p = nct.cdf(-t, df, (delta - upper) / se) - nct.cdf(t, df, (delta - lower) / se)
        return float(max(p, 0.0))
    raise ValueError("hypothesis must be superiority, noninferiority or equivalence")


def continuous_sample_size(
    design: str = "parallel",
    hypothesis: str = "superiority",
    delta: float | None = None,
    sd: float | None = None,
    alpha: float | None = None,
    power: float | None = 0.8,
    n: int | None = None,
    margin: float | None = None,
    lower_margin: float | None = None,
    upper_margin: float | None = None,
    ratio: float = 1.0,
    sides: int = 2,
    scale: str = "difference",
    gmr: float | None = None,
    cv: float | None = None,
) -> CalcResult:
    """Sample size (``n`` omitted) or power (``n`` given) for a continuous endpoint.

    ``scale="ratio"`` works on the log scale from a geometric mean ratio ``gmr`` and
    within-subject ``cv`` (bioequivalence-style); margins are then ratios (e.g. 0.8/1.25).
    """
    if design not in _DESIGNS:
        raise ValueError(f"design must be one of {list(_DESIGNS)}")
    if hypothesis not in ("superiority", "noninferiority", "equivalence"):
        raise ValueError("hypothesis must be superiority, noninferiority or equivalence")
    if alpha is None:
        alpha = 0.05 if (hypothesis == "superiority" and sides == 2) or hypothesis == "equivalence" else 0.025
    _check_prob("alpha", alpha)
    notes: list[str] = []

    if scale == "ratio":
        if gmr is None or cv is None:
            raise ValueError("scale='ratio' needs gmr and cv")
        if gmr <= 0 or cv <= 0:
            raise ValueError("gmr and cv must be > 0")
        delta = math.log(gmr)
        sd = math.sqrt(math.log(1 + cv**2))
        lower_ratio = lower_margin if lower_margin is not None else 0.8
        lo = math.log(lower_ratio)
        hi = None
        if hypothesis == "equivalence":
            hi = math.log(upper_margin if upper_margin is not None else 1 / lower_ratio)
        notes.append(f"log scale: delta=ln({gmr})={delta:.4f}, sd=sqrt(ln(1+CV^2))={sd:.4f}")
    else:
        if delta is None or sd is None:
            raise ValueError("delta and sd are required")
        if lower_margin is not None:
            lo = lower_margin
        elif margin is not None:
            lo = -abs(margin)
        else:
            lo = None
        if upper_margin is not None:
            hi = upper_margin
        elif margin is not None and hypothesis == "equivalence":
            hi = abs(margin)
        else:
            hi = None
    if sd <= 0:
        raise ValueError("sd must be > 0")
    if hypothesis == "noninferiority" and (lo is None or lo >= 0):
        raise ValueError("non-inferiority needs a margin > 0 (H0: delta <= -margin)")
    if hypothesis == "equivalence":
        if hi is None or lo is None or not (lo < hi):
            raise ValueError("equivalence needs margins with lower < upper (e.g. margin=3 for +/-3)")
        if not (lo < delta < hi):
            raise ValueError("true difference must lie inside the equivalence margins")
    if hypothesis == "superiority" and delta == 0:
        raise ValueError("superiority needs a non-zero delta")
    if hypothesis != "superiority":
        sides = 1

    def pw(nn: int) -> float:
        se, df, _ = _se_df(design, nn, sd, ratio)
        return _power_t(hypothesis, delta, se, df, alpha, sides, lo, hi)

    step = 2 if design == "crossover_2x2" else 1
    n_min = 4 if design == "crossover_2x2" else 2

    if n is not None:
        n = int(n)
        if design == "crossover_2x2" and n % 2:
            notes.append(f"n={n} is odd; rounded up to {n + 1} for equal sequences")
            n += 1
        if n < n_min:
            raise ValueError(f"n must be at least {n_min}")
        achieved = pw(n)
        se, df, sizes = _se_df(design, n, sd, ratio)
        outputs = {**sizes, "power": round(achieved, 4), "se": round(se, 5), "df": df}
        solve = "power"
    else:
        if power is None:
            raise ValueError("give either n (to get power) or power (to get n)")
        _check_prob("power", power)
        # Normal-approximation start, then exact search on the t-based power.
        z_a = stats.norm.ppf(1 - (alpha / 2 if (hypothesis == "superiority" and sides == 2) else alpha))
        z_b = stats.norm.ppf(power if hypothesis != "equivalence" else 1 - (1 - power) / 2)
        if hypothesis == "superiority":
            eff = abs(delta)
        elif hypothesis == "noninferiority":
            eff = delta - lo
        else:
            eff = min(delta - lo, hi - delta)
        if eff <= 0:
            raise ValueError("no power possible: true difference does not favour the hypothesis")
        k = {"parallel": 1 + 1 / ratio, "crossover_2x2": 2.0, "paired": 1.0}[design]
        start = max(n_min, int(math.ceil(k * ((z_a + z_b) * sd / eff) ** 2)))
        start += (start % step) if step == 2 else 0
        nn = start
        while nn > n_min and pw(nn - step) >= power:
            nn -= step
        while pw(nn) < power:
            nn += step
            if nn > MAX_N:
                raise ValueError("required n exceeds 200,000 - check the inputs")
        se, df, sizes = _se_df(design, nn, sd, ratio)
        outputs = {**sizes, "power": round(pw(nn), 4)}
        solve = "n"

    method_bits = {
        "superiority": f"{'two' if sides == 2 else 'one'}-sided t-test, exact power from the noncentral t distribution",
        "noninferiority": "one-sided t-test of H0: delta <= -margin, exact power from the noncentral t distribution",
        "equivalence": "two one-sided tests (TOST), power by the noncentral-t approximation (as PowerTOST method='nct')",
    }
    inputs = {
        "design": design, "hypothesis": hypothesis, "solve_for": solve, "alpha": alpha,
        "target_power" if solve == "n" else "n": power if solve == "n" else n,
        "delta": None if scale == "ratio" else delta, "sd": None if scale == "ratio" else sd,
        "gmr": gmr, "cv": cv,
        "lower_margin": (math.exp(lo) if scale == "ratio" else lo) if hypothesis != "superiority" else None,
        "upper_margin": (math.exp(hi) if scale == "ratio" and hi is not None else hi) if hypothesis == "equivalence" else None,
        "allocation_ratio (trt:ctl)": ratio if design == "parallel" else None,
    }
    assumptions = [_DESIGNS[design], "normally distributed outcome (or its log)", "no dropout (inflate separately)"]
    if design == "crossover_2x2":
        assumptions.append("no carryover; period effects removed by the AB/BA analysis")
    return CalcResult(
        calculator="continuous_sample_size",
        inputs={k: v for k, v in inputs.items() if v is not None},
        outputs=outputs,
        method=method_bits[hypothesis],
        assumptions=assumptions,
        notes=notes,
    )


# ==========================================================================
# Binary endpoints: two independent proportions
# ==========================================================================


def _ftu(n: float, k: float, d: float) -> float:
    return n / 4 * (1 + math.sqrt(1 + 2 * (k + 1) / (k * n * d))) ** 2


def _ftu_inverse(n_cc: float, k: float, d: float) -> float:
    """Uncorrected n whose continuity-corrected size equals n_cc."""
    from scipy.optimize import brentq

    if _ftu(1e-6, k, d) >= n_cc:
        return 1e-6
    return float(brentq(lambda n: _ftu(n, k, d) - n_cc, 1e-6, n_cc))


def binary_sample_size(
    p_control: float,
    p_treatment: float,
    hypothesis: str = "superiority",
    alpha: float | None = None,
    power: float | None = 0.8,
    n_control: int | None = None,
    margin: float | None = None,
    ratio: float = 1.0,
    continuity_correction: bool = False,
) -> CalcResult:
    """Two independent proportions (risk difference, treatment - control).

    Superiority: two-sided pooled z-test (Fleiss); optional Fleiss-Tytun-Ury
    continuity correction. Non-inferiority: one-sided Wald test with unpooled
    variance (Blackwelder), H0: p_t - p_c <= -margin.
    """
    _check_prob("p_control", p_control)
    _check_prob("p_treatment", p_treatment)
    if hypothesis not in ("superiority", "noninferiority"):
        raise ValueError("hypothesis must be superiority or noninferiority")
    if alpha is None:
        alpha = 0.05 if hypothesis == "superiority" else 0.025
    k = ratio
    pc, pt = p_control, p_treatment
    d = pt - pc
    notes: list[str] = []

    if hypothesis == "superiority":
        if d == 0:
            raise ValueError("superiority needs p_treatment != p_control")
        za = stats.norm.ppf(1 - alpha / 2)
        pbar = (pc + k * pt) / (1 + k)
        s0 = math.sqrt(pbar * (1 - pbar) * (1 + 1 / k))
        s1 = math.sqrt(pc * (1 - pc) + pt * (1 - pt) / k)

        def pw(nc: int) -> float:
            nc_eff = float(nc)
            if continuity_correction:  # invert the Fleiss-Tytun-Ury correction exactly
                nc_eff = _ftu_inverse(nc, k, abs(d))
            return float(stats.norm.cdf((abs(d) * math.sqrt(nc_eff) - za * s0) / s1))

        if n_control is None:
            _check_prob("power", power)
            zb = stats.norm.ppf(power)
            n_raw = ((za * s0 + zb * s1) / abs(d)) ** 2
            if continuity_correction:
                n_raw = _ftu(n_raw, k, abs(d))
                notes.append("Fleiss-Tytun-Ury continuity correction applied")
            nc = math.ceil(n_raw)
        method = "two-sided pooled z-test for two proportions (Fleiss formula)"
    else:
        if margin is None or margin <= 0:
            raise ValueError("non-inferiority needs margin > 0 (on the risk-difference scale)")
        if d + margin <= 0:
            raise ValueError("true difference is not better than -margin: power cannot reach the target")
        za = stats.norm.ppf(1 - alpha)
        var1 = pc * (1 - pc) + pt * (1 - pt) / k

        def pw(nc: int) -> float:
            return float(stats.norm.cdf((d + margin) / math.sqrt(var1 / nc) - za))

        if n_control is None:
            _check_prob("power", power)
            zb = stats.norm.ppf(power)
            nc = math.ceil((za + zb) ** 2 * var1 / (d + margin) ** 2)
        method = "one-sided Wald z-test of H0: p_t - p_c <= -margin, unpooled variance (Blackwelder)"
        if continuity_correction:
            notes.append("continuity correction is not applied for non-inferiority")

    if n_control is not None:
        nc = int(n_control)
        solve = "power"
    else:
        solve = "n"
    nt = math.ceil(k * nc)
    outputs = {"n_control": nc, "n_treatment": nt, "n_total": nc + nt, "power": round(pw(nc), 4)}
    inputs = {"hypothesis": hypothesis, "solve_for": solve, "p_control": pc, "p_treatment": pt, "alpha": alpha,
              "margin": margin, "allocation_ratio (trt:ctl)": k}
    if solve == "n":
        inputs["target_power"] = power
    return CalcResult(
        calculator="binary_sample_size",
        inputs={k_: v for k_, v in inputs.items() if v is not None},
        outputs=outputs,
        method=method,
        assumptions=["independent subjects (one observation per subject)", "large-sample normal approximation",
                     "no dropout (inflate separately)"],
        notes=notes,
    )


# ==========================================================================
# Ophthalmic: correlated eyes, dropout
# ==========================================================================


def correlated_eyes_adjustment(
    n_independent_eyes: int | None = None,
    n_subjects: int | None = None,
    eyes_per_subject: float = 2.0,
    icc: float = 0.5,
) -> CalcResult:
    """Design effect when both eyes of a subject receive the SAME treatment.

    DE = 1 + (m - 1) * ICC, with m eyes per subject. Give ``n_independent_eyes``
    (sample size computed as if eyes were independent) to get the subjects
    needed, or ``n_subjects`` to get the effective number of independent eyes.
    """
    if not (0 <= icc <= 1):
        raise ValueError("icc must be in [0, 1]")
    if eyes_per_subject < 1 or eyes_per_subject > 2:
        raise ValueError("eyes_per_subject must be between 1 and 2 (can be an average, e.g. 1.8)")
    m = eyes_per_subject
    de = 1 + (m - 1) * icc
    if n_independent_eyes is not None:
        eyes = math.ceil(n_independent_eyes * de)
        subjects = math.ceil(eyes / m)
        outputs = {"design_effect": round(de, 4), "eyes_needed": eyes, "subjects_needed": subjects}
        inputs = {"n_independent_eyes": n_independent_eyes}
    elif n_subjects is not None:
        eff = n_subjects * m / de
        outputs = {"design_effect": round(de, 4), "total_eyes": round(n_subjects * m, 1),
                   "effective_independent_eyes": round(eff, 1)}
        inputs = {"n_subjects": n_subjects}
    else:
        raise ValueError("give n_independent_eyes or n_subjects")
    inputs.update({"eyes_per_subject": m, "icc": icc})
    return CalcResult(
        calculator="correlated_eyes_adjustment",
        inputs=inputs,
        outputs=outputs,
        method="cluster design effect DE = 1 + (m - 1) x ICC (Kish); eyes are clusters within subjects",
        assumptions=["both eyes of a subject receive the same treatment (subject-level randomisation)",
                     "common intra-subject (inter-eye) correlation ICC"],
        notes=["If the two eyes receive DIFFERENT treatments (contralateral design), use the paired design "
               "in continuous_sample_size instead - between-eye correlation then increases efficiency."],
    )


def inflate_for_dropout(n: int, dropout_rate: float, multiple_of: int = 1) -> CalcResult:
    """Enrol n / (1 - dropout) subjects, rounded up to a multiple (e.g. sequences or block size)."""
    if not (0 <= dropout_rate < 1):
        raise ValueError("dropout_rate must be in [0, 1)")
    if multiple_of < 1:
        raise ValueError("multiple_of must be >= 1")
    raw = n / (1 - dropout_rate)
    enrol = math.ceil(raw - 1e-9)
    enrol = int(math.ceil(enrol / multiple_of) * multiple_of)
    return CalcResult(
        calculator="inflate_for_dropout",
        inputs={"evaluable_n": n, "dropout_rate": dropout_rate, "round_to_multiple_of": multiple_of},
        outputs={"enrol_n": enrol, "expected_evaluable": round(enrol * (1 - dropout_rate), 1)},
        method="N_enrol = ceil(N / (1 - dropout)), rounded up to the requested multiple",
        assumptions=["dropouts are non-informative and contribute no data to the primary analysis"],
    )


# ==========================================================================
# Multiplicity
# ==========================================================================

_ADJ_METHODS = ("bonferroni", "holm", "hochberg", "bh", "fixed_sequence")


def adjust_pvalues(pvalues: list[float], method: str = "holm", alpha: float = 0.05,
                   labels: list[str] | None = None) -> CalcResult:
    """Multiplicity-adjusted p-values and rejections.

    ``fixed_sequence`` tests in the order given, each at full alpha, stopping at
    the first non-rejection (adjusted p = running maximum).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a non-empty list of numbers in [0, 1]")
    if method not in _ADJ_METHODS:
        raise ValueError(f"method must be one of {_ADJ_METHODS}")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    if method == "bonferroni":
        adj = np.minimum(p * m, 1)
    elif method == "holm":
        ps = p[order] * (m - np.arange(m))
        adj[order] = np.minimum(np.maximum.accumulate(ps), 1)
    elif method == "hochberg":
        ps = p[order] * (m - np.arange(m))
        adj[order] = np.minimum(np.minimum.accumulate(ps[::-1])[::-1], 1)
    elif method == "bh":
        ps = p[order] * m / (np.arange(m) + 1)
        adj[order] = np.minimum(np.minimum.accumulate(ps[::-1])[::-1], 1)
    elif method == "fixed_sequence":
        adj = np.maximum.accumulate(p)
    reject = (adj <= alpha).tolist()
    labels = labels or [f"H{i + 1}" for i in range(m)]
    method_text = {
        "bonferroni": "Bonferroni: p x m (controls FWER under any dependence)",
        "holm": "Holm step-down (controls FWER under any dependence)",
        "hochberg": "Hochberg step-up (controls FWER under independence or positive dependence, e.g. PRDS)",
        "bh": "Benjamini-Hochberg step-up (controls the false discovery rate, not FWER)",
        "fixed_sequence": "fixed-sequence (hierarchical) testing in the order given, each at full alpha",
    }[method]
    return CalcResult(
        calculator="adjust_pvalues",
        inputs={"pvalues": [float(x) for x in p], "labels": labels, "method": method, "alpha": alpha},
        outputs={"adjusted": [round(float(x), 6) for x in adj], "reject": reject,
                 "rejected": [l for l, r in zip(labels, reject) if r]},
        method=method_text,
        assumptions=["p-values are valid unadjusted p-values for the stated hypotheses"],
    )


CALCULATORS = {
    "continuous_sample_size": continuous_sample_size,
    "binary_sample_size": binary_sample_size,
    "correlated_eyes_adjustment": correlated_eyes_adjustment,
    "inflate_for_dropout": inflate_for_dropout,
    "adjust_pvalues": adjust_pvalues,
}
