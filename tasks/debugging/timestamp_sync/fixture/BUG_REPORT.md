# Bug: camera frames near the end of a lidar log are never associated

Observed in the fusion replay tool:

    >>> match_nearest([10.04], [9.0, 10.0], tolerance=0.05)
    [None]

Expected `[1]` (10.0 is within 0.05 s of 10.04). Frames in the middle of the log look fine.
Please reproduce, find the root cause, make a minimal fix, and add a regression test.
Also check the function honours its documented contract.
