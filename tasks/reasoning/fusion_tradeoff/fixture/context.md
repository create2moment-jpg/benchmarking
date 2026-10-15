# Context: highway ACC target selection

Sensors on the test vehicle:

| Sensor        | Rate  | Typical latency | Strengths                         | Weaknesses                      |
|---------------|-------|-----------------|-----------------------------------|---------------------------------|
| Front radar   | 20 Hz | 30 ms           | Range rate, works in rain/fog      | Poor lateral accuracy, clutter  |
| Front camera  | 30 Hz | 80 ms (variable)| Classification, lane assignment    | Range at distance, glare, night |
| Roof lidar    | 10 Hz | 50 ms           | Accurate 3-D shape and position    | Heavy rain, cost, compute       |

Constraints: the ACC controller needs a fused lead-vehicle estimate every 50 ms with end-to-end
latency below 150 ms; the perception ECU has a fixed CPU budget and no spare GPU. The camera and
radar suppliers each deliver object lists (tracks); raw radar detections and lidar points are
also available on the vehicle bus.
