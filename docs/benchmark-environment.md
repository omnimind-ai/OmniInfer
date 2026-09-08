# Benchmark environment evidence

Campaign runners can import `scripts/benchmark_environment.py` for prospective
environment gates. These helpers neither alter device policy nor publish data.
Keep raw observations, source/binary hashes, phase markers, and entire excluded
cohorts. Do not backfill old measurements with new observations.

## Windows GPU counters

Run `scripts/benchmark_windows_gpu.ps1 -OutputPath <new-jsonl-file>` continuously
from before loading until after the final request. For a bounded collector check,
pass `-MaxSamples 3`; the default collects continuously. Record UTC phase boundaries
for loading, ready, QA, calibration, warmup and scored requests separately.
Allow the predeclared post-readiness settling period before QA (the September 8
replacement cohort used five seconds). It is part of both comparison protocols.

The collector retains zero and invalid counters, actual consecutive timestamp
boundaries, and an unknown start for its first sample. Parse the ISO timestamps
as timezone-aware times and call `foreign_gpu_activity` for each interval and
protected window. Use the actual owned process IDs, not process names. An
interval crossing a boundary is ambiguous; it does not establish when activity
occurred. Foreign activity above the threshold still excludes that cohort;
ambiguity is not permission to discard a sample or exempt PID 4. Missing or
invalid in-window counters fail closed. Preserve the returned attribution.

## Android display and prelaunch temperature

After asset verification and before starting the runtime, call
`wait_for_prelaunch`. Its ownership callback must re-read the device lease and
check for foreign inference/download/checksum jobs. Supply a timestamped battery
observation and append every sample through `record`. A timeout or changed lease
is a preparation failure, not a measured throughput result. No inference starts
inside this helper. Do not change the temperature threshold after seeing data.

Save the original display state before changing it. Declare the test state in
the job manifest. Before QA and every request, and again after each request,
call `android_display_snapshot(shell)` and `verify_display`, appending all raw
samples to the cohort's journal. Manual brightness, brightness mode, timeout
and wakefulness must match the declaration. A mismatch excludes the cohort;
do not silently overwrite a user change. Snapshotting adds observation overhead,
so place it outside measured requests and use the same protocol on both sides.

Only the final lease owner restores original settings. Restore manual brightness
and verify it before restoring the original automatic mode, if applicable;
automatic brightness may subsequently change. Verify wakefulness with bounded
polling because `Dozing` after a sleep event is transitional. `verify_display`
checks this without changing settings. Historical evidence gaps remain gaps.

Validate the helpers with:

```sh
python3 -m unittest discover -s tests -p test_benchmark_environment.py
```
