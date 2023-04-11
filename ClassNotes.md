# Class Notes
---------------------------------------------------------------
## Tabular Q-Learning problems
- State space explosion

## Naive DQN problems
- Observation: Learning curve. Rises till epsilon=0.2. 2.5K episodes. Avg reward=0.2. Then after epsilon reaches minimum = 0.001, slowly starts rising at 4.5K, max. at 7K = avg.reward=1.0; THEN catastrophically falls, then rises to max. about 0.8-0.9/falls to zero in cycles. Unstable

** Problems **
- Learning from one example at a time. Agent sees 1000 of steps but discarded every time. Starts again.
- Enormous parameter space
- When we start an episode: i.e. When epsilon is large - can explore action space
- When epsilon is small (0.01), gets caught in local minimas!!! Classic curve.
- Using same network to **evaluate** max action as well as **choose** max action.
- Updated EVERY step - so "chasing a moving target"!
- $max_a Q(s',a_max)$ part. Is biased toward max.
- Possible to learn with small space (cart-pole=4 space)
-  