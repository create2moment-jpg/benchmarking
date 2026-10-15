CREATE TABLE readings (sensor_id VARCHAR NOT NULL, ts TIMESTAMP NOT NULL, value DOUBLE);
INSERT INTO readings VALUES
  ('lidar_roof',  TIMESTAMP '2026-01-01 00:00:00', 1.0),
  ('lidar_roof',  TIMESTAMP '2026-01-01 00:00:05', 1.0),   -- exactly 5 s: not a gap
  ('lidar_roof',  TIMESTAMP '2026-01-01 00:00:11.5', NULL), -- 6.5 s gap; NULL value still a heartbeat
  ('lidar_roof',  TIMESTAMP '2026-01-01 00:00:12', 2.0),
  ('radar_front', TIMESTAMP '2026-01-01 00:00:20', 1.0),   -- rows deliberately out of order
  ('radar_front', TIMESTAMP '2026-01-01 00:00:00', 1.0),
  ('radar_front', TIMESTAMP '2026-01-01 00:00:02', 1.0),
  ('radar_front', TIMESTAMP '2026-01-01 00:00:02', 1.5),   -- duplicate timestamp
  ('radar_front', TIMESTAMP '2026-01-01 00:00:10', 1.0),   -- 8 s gap
  ('radar_front', TIMESTAMP '2026-01-01 00:01:00', 1.0),   -- 40 s gap after :20
  ('cam_left',    TIMESTAMP '2026-01-01 00:00:00', 1.0),   -- single row: no gaps
  ('radar_rear',  TIMESTAMP '2025-12-31 23:59:58', 1.0),   -- crosses midnight/year
  ('radar_rear',  TIMESTAMP '2026-01-01 00:00:04', 1.0);
