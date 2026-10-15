# Recommendation: track-level (late) fusion of radar and camera, lidar as optional input

Assumption: supplier object lists are available at rate with per-object covariance.

**Why late fusion.** The ECU has a fixed CPU budget and no spare GPU; early/low-level fusion of
raw lidar points and radar detections would need point-cloud processing we cannot afford within
the 50 ms cycle and 150 ms latency budget. Track-level fusion of supplier object lists is cheap,
modular and lets each supplier keep its sensor-specific processing. We lose some information
(low-level evidence is discarded), so we add raw radar detections near the lead vehicle as a
hybrid refinement where range-rate matters most.

**Timestamps and latency.** Every object is stamped with its measurement time, not arrival time.
The fusion predicts tracks to each measurement timestamp; out-of-sequence measurements (the 80 ms
camera arrives after newer radar data) are handled with a short state history (retrodiction) or,
cheaper, by buffering up to ~90 ms and processing in timestamp order, which keeps end-to-end
latency below 150 ms.

**Coordinate frames and calibration.** All objects are transformed into the vehicle frame using
extrinsic calibration; calibration errors are monitored online (radar/camera bearing residuals)
and inflate covariance when drift is detected.

**Uncertainty and correlation.** Supplier tracks are filtered estimates whose errors are
correlated over time and with each other through shared motion models, so naive Kalman fusion
would double count information. Use covariance intersection (or track-to-track fusion with
de-correlation) to stay consistent.

**Compute.** Track-level association is O(tracks^2) on tens of objects, well within budget.

**Degraded modes.** On camera failure (glare, night) fall back to radar-only with reduced ACC
capability and larger following distance; on radar outage, camera-only with speed limitation;
the HMI is informed. Sensor health is checked via timeouts and plausibility.

**Validation.** Replay of recorded drives with ground truth from lidar, metrics: lead-vehicle
selection accuracy, range/range-rate error, latency distribution; ablation with each sensor
removed; fault injection for delays and dropouts.
