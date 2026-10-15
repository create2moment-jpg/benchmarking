WITH distinct_ts AS (
    SELECT DISTINCT sensor_id, ts FROM readings
),
ordered AS (
    SELECT
        sensor_id,
        LAG(ts) OVER (PARTITION BY sensor_id ORDER BY ts) AS gap_start,
        ts AS gap_end
    FROM distinct_ts
)
SELECT
    sensor_id,
    gap_start,
    gap_end,
    CAST(EPOCH(gap_end) - EPOCH(gap_start) AS DOUBLE) AS gap_seconds
FROM ordered
WHERE gap_start IS NOT NULL
  AND EPOCH(gap_end) - EPOCH(gap_start) > 5
ORDER BY sensor_id, gap_start;
