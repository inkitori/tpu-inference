# Windowed draft attention (Lever A) acceptance check

Mean accept length = 1 + accepted/drafts (includes bonus tok; max 8 for 7 spec tok).
Baseline (full ctx) steady-state ~6.0-6.7 at c=32. Threshold: >=4 for win.

## W=256 (DFLASH_DRAFT_WINDOW=256) -- came up CLEAN at c=32, no OOM/crash
Steady-state fat warm windows (24-way concurrency, drafted>>1000):
- drafted=15960 mean=2.44 perpos=[0.65,0.38,0.20,0.11,0.06,0.03,0.02]
- drafted=15379 mean=2.59
- drafted=4081  mean=2.56
- drafted=1988  mean=2.33
Tail (thin, low conc): ~3.1-3.2.
=> W=256 steady-state mean accept ~2.4-2.6. WAY below 4. FAIL.

## W=512 -- came up CLEAN at c=32, no OOM/crash
Steady-state fat warm windows:
- drafted=16520 mean=2.61
- drafted=16205 mean=2.48
- drafted=5215  mean=2.38
Tail thin: ~2.97.
=> W=512 steady-state mean accept ~2.4-2.6. IDENTICAL to W=256. Larger window did NOT recover. FAIL.

## VERDICT
Both W=256 and W=512 give mean accept ~2.4-2.6 vs baseline ~6.0-6.7 (threshold >=4).
Windowed draft attention FAILS the acceptance bar. A ~6x-cheaper draft does NOT win
when accept ~2.5 (projection needs accept >=4 to beat target-only B=3752 tok/s).
Both windowed serves RAN CLEAN at c=32 (no crash/OOM) -- the windowed path is runtime-correct,
just not accuracy-viable at these window sizes.
