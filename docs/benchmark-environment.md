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

For a prospective Android campaign, `scripts/benchmark_android_monitor.py`
provides `render_monitor(pid, binary, model)`. It renders the regular one-hertz
process/frequency/battery collector for an owned runtime and absolute shell-safe
campaign paths. Save the rendered script with the cohort, push it to the owned
work directory and start it on the device with output captured to a file.

The existing battery observation also enforces the declared 40°C ceiling during
warmups and in-flight requests. An observation above 40°C or a missing/invalid
sensor emits `SAFETY_STOP`; SIGTERM is sent only when the PID's command line still
contains the exact expected binary and model arguments. Preserve the failed
cohort and raw reading, and clean up the owned forward/runtime. Do not interpret
a partial response as a benchmark result. The guarantee applies to observed
samples, not temperatures between samples; the monitor never changes device
policy or creates heat. Freeze the same monitor for both sides of a comparison.


## Android CPU stability

Declare warmup count before collecting a cohort. Battery temperature alone does
not establish a steady CPU frequency: separate CPU clusters may transition to
lower clocks while the battery remains within the thermal gate. Keep the complete
failed cohort and correlate frequency observations with request boundaries.
Whole-request or one-hertz samples cannot locate an event within subsecond
prefill; use a separate diagnostic with sufficient temporal resolution and keep
its observation overhead out of published performance results.

For MiniCPM5-2B F16 on SM8850 with official `64e9bceb2`, PP512/TG128,
ctx2048, eight threads and strict affinity `ff`, issue #257's single predeclared
replacement used three warmups followed by all three formal rounds. The PP
population CV fell from 5.955% to 0.1056%; TG CV fell from 8.969% to 2.0578%.
That configuration's measured means were 52.0134 and 10.5293 tokens/s. This is a
validated campaign warmup remedy, not a runtime or system frequency-policy fix.
The original one-warmup cohort remains excluded. Do not generalize three warmups
to other devices or use it as permission to retry an unrelated failing cell.

Both backends in a comparison must use the same declared warmup protocol.
Preserve the 5% population CV gate, fixed model/runtime, token counts, cache
policy, all scored rounds and excluded attempts. If a prospectively declared
corrective cohort still fails, retain the gap rather than selecting stable
rounds, cooling between scored requests or relaxing the threshold.

Validate the helpers with:

```sh
python3 -m unittest discover -s tests -p test_benchmark_environment.py
```
