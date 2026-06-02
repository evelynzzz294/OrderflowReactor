# OrderflowReactor
Reconstructing the MNQ limit order book from market-by-order data and testing a queueing-theoretic (PDE) model of short-horizon price movement

Short-horizon price direction is one of the central objects in market microstructure and market-making. A simple but influential signal is queue imbalance, which accounts for the relative size of the best bid and best ask queues. This is used here to compute **P(up before down | book state)**, the probability that the mid-price moves up before it moves down.  Queue-based models provide a theoretically grounded estimate of that probability from observable liquidity, and understanding where those models succeed and fail quantifies how much information top-of-book liquidity alone actually carries. This project reconstructs the CME MNQ limit order book from event-level market-by-order data, measures the empirical relationship between queue imbalance and future price direction, and tests a parameter-free queueing/PDE model against it.

# Key Findings
- Reconstructed **100M+** order-book events from CME MNQ futures into a validated L3 book.
- Generated **88M+** first-passage up/down labels across four trading sessions.
- **Reproduced** the known monotonic relationship between queue imbalance and the
direction of the next price move (Gould–Bonart 2015), here on CME futures.
- Compared the empirical curve against a **parameter-free queueing/PDE model** (the
hitting-probability / harmonic-function prediction).
- The model gets the **direction and the 50/50 crossing point** right, but systematically
**overstates directional certainty** — mean absolute deviation 0.14, up to 0.24 in
extreme-imbalance regimes.
- The gap is **stable across all four sessions** (per-bin std < 0.011 in populated bins).
