# Results ledger — 2026-10-29

**Hypothesis:** Circulation protects accessible past bits; local multiplication makes historical XOR linearly readable.

**Method:** 5 random seeds; 16D, 12-step binary antipodal streams; 1,200 train + 600 independent held-out examples per seed; Gaussian input noise SD 0.01; input/noise shared across architectures within seeds; linear ridge + intercept; fixed reservoir matrices, not trained. Run benchmark.py --seeds 5 --train 1200 --test 600 --json results.json for seedwise output.

**Observed mean scores:**
- Signed shift + gate: lag4 94.7%, lag6 75.9%, XOR(lags 1,2) 87.0%, XOR(lags 2,3) 83.3%.
- Signed shift without gate: lag4 100%, lag6 97.3%, XOR 52.8% / 50.9%.
- Cyclic + gate: XOR(lags 1,2) 87.2%; random orthogonal + gate: 87.5%.
- Erase history before last two stimuli: older XOR ~48–50% (chance).
- The **ungated exactly linear** signed shift recalls past individual symbols perfectly through lag6 but cannot perform XOR linearly.

**Probe:** Write either sign of code at t=0, hold zero input, probe with either sign; four-way difference in raw states. Signed + gate interaction norm 2.5896 at delay 0, 0.4910 at delay 16, 0.0088 at 64; ungated signed shift exactly zero up to numerical precision. Nonzero interactions also occur with ordinary cyclic and identity gated recurrences.

**Falsified:** No special advantage of Möbius sign reversal in tested tasks. Circular recurrence alone does not make XOR linearly available under strict antipodal inputs.

**Unresolved:** Physical waves, biology, energy conservation, longer-term capacity, trained competitive architecture benchmarks, multiple modalities, generalization across encoding channels. High lag recall is expected from a well-conditioned synthetic 16D rotated input code; not a neuronal mechanism.

**Next:** Pre-register full-turn sign test; held-out amplitude/delay/noise and multichannel sequences; fair parameterized competitors; replace engineered rotation with a physical oscillator/field dynamics and an energy ledger.
