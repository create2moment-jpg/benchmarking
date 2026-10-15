-- Plausible but wrong: ignores re-ingestion and drops sensors without readings.
SELECT r.sensor_id, s.kind, r.ts, r.value
FROM readings r
JOIN sensors s ON s.sensor_id = r.sensor_id
WHERE s.active = 1 AND r.value IS NOT NULL
  AND r.ts = (SELECT MAX(ts) FROM readings r2 WHERE r2.sensor_id = r.sensor_id AND r2.value IS NOT NULL)
GROUP BY r.sensor_id
ORDER BY r.sensor_id;
