-- Schema (SQLite). The evaluator seeds a different, larger dataset with the same schema.
CREATE TABLE sensors (
    sensor_id TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,          -- 'radar' | 'lidar' | 'camera'
    active    INTEGER NOT NULL        -- 1 = active, 0 = decommissioned
);
CREATE TABLE readings (
    id          INTEGER PRIMARY KEY,
    sensor_id   TEXT NOT NULL REFERENCES sensors(sensor_id),
    ts          INTEGER NOT NULL,     -- measurement time, epoch milliseconds
    value       REAL,                 -- NULL when the sensor reported a dropout
    ingested_at INTEGER NOT NULL      -- when the row was (re-)ingested, epoch milliseconds
);
