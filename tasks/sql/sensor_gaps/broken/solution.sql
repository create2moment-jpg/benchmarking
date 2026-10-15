-- Plausible but wrong: no dedup of identical timestamps, >= instead of >, ignores NULL heartbeats.
SELECT sensor_id, gap_start, gap_end, CAST(EPOCH(gap_end) - EPOCH(gap_start) AS DOUBLE) AS gap_seconds
FROM (
    SELECT sensor_id, LAG(ts) OVER (PARTITION BY sensor_id ORDER BY ts) AS gap_start, ts AS gap_end
    FROM readings WHERE value IS NOT NULL
)
WHERE gap_start IS NOT NULL AND EPOCH(gap_end) - EPOCH(gap_start) >= 5
ORDER BY sensor_id, gap_start;
