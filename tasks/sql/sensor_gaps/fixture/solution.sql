-- TODO: report heartbeat gaps longer than 5 seconds per sensor.
SELECT sensor_id, MIN(ts) AS gap_start, MAX(ts) AS gap_end,
       EXTRACT(EPOCH FROM MAX(ts) - MIN(ts)) AS gap_seconds
FROM readings
GROUP BY sensor_id;
