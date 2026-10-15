-- Latest reading per sensor. (Reported buggy: see task description.)
SELECT r.sensor_id, s.kind, MAX(r.ts) AS ts, r.value
FROM readings r
JOIN sensors s ON s.sensor_id = r.sensor_id
GROUP BY r.sensor_id;
