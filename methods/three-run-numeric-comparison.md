# Independent three-run numerical comparison

Computed directly from the three saved `comparison.json` files and training CSVs. The supplemental comparison is kept separate from the official classroom evaluation.

## Official five-game results

| Fresh run | Untrained mean | Trained mean | Trained median | Change from its own baseline |
|---|---:|---:|---:|---:|
| 50 episodes | 492 | 800 | 510 | +308 / +62.60% |
| 200 episodes | 492 | 534 | 510 | +42 / +8.54% |
| 500 episodes, decay and new seed | 218 | 450 | 480 | +232 / +106.42% |

The final 500-run improved all five official games against its own untrained model. It nevertheless scored **350 points below the 50-run mean (−43.75%)** and **84 below the 200-run mean (−15.73%)**. Against 50 episodes, two seeds improved and three worsened; against 200, three improved and two worsened. There were no ties. These are comparisons of the same official seed set, not evidence identifying which training change caused the difference.

The 500-run's paired changes versus 50 were **−20, +70, −1,550, +50, −300**. Versus 200 they were **+70, +70, −170, +50, −440**, in seed order 101/202/303/404/505.

## Supplemental five-game results

The final 500-run scored **910, 420, 830, 370, 1,900**, mean **886**, versus baseline mean **252**: **+634 points / +251.59%**. All five improved. The median increased from **240 to 830**. These additional seeds produced stronger results than the official set, illustrating sensitivity to the evaluated games; they must not replace or be pooled into the official mean of 450. No official or supplemental baseline/final evaluation game reached the time limit.

## Measured training-curve summary

| Run | First 25 training-game mean | Final 25 mean | Highest complete 25-game rolling mean | Last episode of peak window |
|---|---:|---:|---:|---:|
| 50 | 637.2 | 673.6 | 801.2 | 37 |
| 200 | 553.2 | 736.4 | 847.6 | 90 |
| 500 | 584.0 | 637.6 | 891.2 | 170 |

The 500-run's rolling score improved above its starting window, fluctuated, and finished below its earlier peak. Its successive 125-episode block means were **633.84, 686.00, 653.76, and 621.84**. Individual scores ranged from **140 to 2,220**. None of its training games reached the time limit.

The average of finite per-episode mean losses in the first 25 games was **0.03379**, versus **0.09041** in the final 25. This is an equal-episode summary of the logged losses, not a loss average weighted by update count. Lower rates did not imply lower measured losses; loss values alone do not establish gameplay quality or the causal effect of decay.

## Budget, audit, and interpretation

The 500-run completed **500 episodes, 292,152 decisions, and 72,789 updates** in **1,150.27 training seconds** including periodic demos (approximately **19 minutes 10 seconds**). Initial/final learning rates were **0.0001 / 0.0000512**; every row and boundary matched the intended schedule.

The full audit passed **5,507 checks, zero errors, one warning**. The downloaded notebook preserves **59 outputs and 24 images**, and all **28 code-cell sources** exactly match the prepared notebook. Its Colab export lost every execution counter and contains no `executionInfo`; no counters were reconstructed. The explicit opt-in audit reconciled all 500 printed training rows, progress counters, periodic images, final tables, and saved artifacts. Individual cell counters remain unverifiable from the export.

Confidence is **high** in these recorded values and calculations, **low** in a general ranking of training methods. The 500-run changed episode budget, training seed, and learning-rate schedule together. The earlier 200-run also reused official evaluation seeds 101/202 as training reset seeds, whereas the 50- and 500-run reset ranges excluded their evaluation sets. Repeated reset seeds do not imply identical trajectories. One training realization and small evaluation sets cannot identify one causal improvement or failure mechanism.
