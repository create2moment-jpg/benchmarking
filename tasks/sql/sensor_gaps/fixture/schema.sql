-- DuckDB schema. The evaluator seeds a different, larger dataset.
CREATE TABLE readings (
    sensor_id VARCHAR NOT NULL,
    ts        TIMESTAMP NOT NULL,   -- measurement time (UTC)
    value     DOUBLE                -- may be NULL (dropout); still counts as a heartbeat
);
