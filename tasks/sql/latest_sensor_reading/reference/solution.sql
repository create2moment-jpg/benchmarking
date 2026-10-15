-- Latest non-NULL reading per active sensor; re-ingested duplicates resolved by ingested_at.
WITH ranked AS (
    SELECT
        r.sensor_id,
        r.ts,
        r.value,
        ROW_NUMBER() OVER (
            PARTITION BY r.sensor_id
            ORDER BY r.ts DESC, r.ingested_at DESC, r.id DESC
        ) AS rn
    FROM readings r
    WHERE r.value IS NOT NULL
)
SELECT s.sensor_id, s.kind, k.ts, k.value
FROM sensors s
LEFT JOIN ranked k ON k.sensor_id = s.sensor_id AND k.rn = 1
WHERE s.active = 1
ORDER BY s.sensor_id;
