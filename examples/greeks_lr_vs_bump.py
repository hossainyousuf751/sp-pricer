"""M5: likelihood-ratio vs bump-and-reprice Greeks on the README's worst-of Phoenix.

Bump-and-reprice gives no standard error, so its spread is measured over independent seeds: the
standard deviation of 20 runs is the standard error of one run. Both use 200,000 paths per run.
"""

import time

import numpy as np

from sppricer import Market, PhoenixAutocallable, Underlying, bump_greeks, lr_greeks

market = Market(
    underlyings=(
        Underlying("STOCK_A", spot=100.0, vol=0.25, div_yield=0.02),
        Underlying("STOCK_B", spot=50.0, vol=0.30, div_yield=0.015),
    ),
    rate=0.04,
    correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
)
note = PhoenixAutocallable(
    initial_levels=(100.0, 50.0), observation_times=tuple(0.25 * k for k in range(1, 13)), coupon_rate=0.02,
    coupon_barrier=0.70, autocall_barrier=1.00, protection_barrier=0.60, memory=True, first_autocall_index=3,
)
N, SEEDS = 200_000, range(20)

t0 = time.time()
bumps = [bump_greeks(note, market, n_paths=N, seed=s) for s in SEEDS]
t_bump = (time.time() - t0) / len(SEEDS)
t0 = time.time()
lrs = [lr_greeks(note, market, n_paths=N, seed=s) for s in SEEDS]
t_lr = (time.time() - t0) / len(SEEDS)

print(f"Worst-of Phoenix, {N:,} paths per run, {len(SEEDS)} seeds; time per run: bump {t_bump:.1f}s, LR {t_lr:.1f}s\n")
print(f"{'':10}{'Greek':7}{'bump mean':>12}{'bump SE':>10}{'LR mean':>12}{'LR SE':>10}{'SE ratio':>10}")
for i, u in enumerate(market.underlyings):
    for greek in ("delta", "gamma", "vega"):
        b = np.array([getattr(g, greek)[i] for g in bumps])
        lr = np.array([getattr(g, greek)[i] for g in lrs])
        lr_se = np.mean([getattr(g, greek + "_se")[i] for g in lrs])
        b_se = b.std(ddof=1)
        print(f"{u.name:10}{greek:7}{b.mean():12.5f}{b_se:10.5f}{lr.mean():12.5f}{lr_se:10.5f}{b_se / lr_se:10.1f}")
    lr_gamma = np.array([g.gamma[i] for g in lrs])
    print(f"{'':10}(LR gamma: spread over seeds {lr_gamma.std(ddof=1):.5f} vs its reported SE "
          f"{np.mean([g.gamma_se[i] for g in lrs]):.5f})")
