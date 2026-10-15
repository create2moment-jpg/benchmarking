CREATE TABLE sensors (sensor_id TEXT PRIMARY KEY, kind TEXT NOT NULL, active INTEGER NOT NULL);
CREATE TABLE readings (id INTEGER PRIMARY KEY, sensor_id TEXT NOT NULL, ts INTEGER NOT NULL, value REAL, ingested_at INTEGER NOT NULL);
INSERT INTO sensors VALUES
  ('cam_left',    'camera', 1),
  ('lidar_roof',  'lidar',  1),
  ('radar_front', 'radar',  1),
  ('radar_rear',  'radar',  0),
  ('radar_side',  'radar',  1);
INSERT INTO readings VALUES
  -- radar_front: latest ts is a dropout (NULL) -> use previous non-null; ts 3000 re-ingested
  (1,  'radar_front', 1000, 10.0, 1100),
  (2,  'radar_front', 3000, 30.0, 3100),
  (3,  'radar_front', 3000, 31.5, 3900),
  (4,  'radar_front', 4000, NULL, 4100),
  -- lidar_roof: plain latest
  (5,  'lidar_roof',  2000, 5.0,  2100),
  (6,  'lidar_roof',  2500, 5.5,  2600),
  -- cam_left: only dropouts -> appears with NULL ts/value
  (7,  'cam_left',    1000, NULL, 1100),
  (8,  'cam_left',    2000, NULL, 2100),
  -- radar_rear: inactive -> excluded even though it has the newest data
  (9,  'radar_rear',  9000, 99.0, 9100),
  -- radar_side: out-of-order ingestion; latest ts arrives first, older re-ingest later
  (10, 'radar_side',  5000, 50.0, 5100),
  (11, 'radar_side',  4000, 40.0, 9900),
  (12, 'radar_side',  5000, 49.0, 5050);
