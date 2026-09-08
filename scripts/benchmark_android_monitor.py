"""Render a one-hertz monitor tied to an owned Android runtime.

Only canonical shell-safe campaign paths are accepted. The caller preserves
stdout/stderr and treats a SAFETY_STOP as a failed cohort, never a valid result.
"""
import re

_TEMPLATE = r'''
#!/system/bin/sh
while kill -0 PID_VALUE 2>/dev/null; do
 echo SAMPLE $(date +%s)
 cat /proc/loadavg /proc/PID_VALUE/status 2>/dev/null
 ps -A -o PID,NAME
 for p in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do echo FREQ $p $(cat $p 2>/dev/null); done
 battery_reading=$(dumpsys battery)
 printf '%s\n' "$battery_reading"
 temperature=$(printf '%s\n' "$battery_reading" | sed -n 's/^ *temperature: *//p')
 stop_reason=
 case "$temperature" in ''|*[!0-9]*) stop_reason=INVALID_TEMPERATURE;;
 *) if [ "$temperature" -gt 400 ]; then stop_reason=TEMPERATURE_ABOVE_40C; fi;; esac
 if [ -n "$stop_reason" ]; then
  echo SAFETY_STOP "$stop_reason"
  owned_command=$(tr '\000' ' ' < /proc/PID_VALUE/cmdline)
  case " $owned_command " in *" MODEL_BIN_VALUE "*" MODEL_PATH_VALUE "*) kill -TERM PID_VALUE;; esac
  exit 2
 fi
 sleep 1
done
'''


def render_monitor(pid: int, binary: str, model: str) -> str:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise ValueError("expected an owned runtime PID greater than one")
    for path in (binary, model):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", path):
            raise ValueError("monitor requires absolute shell-safe campaign paths")
    return (_TEMPLATE.lstrip("\n").replace("PID_VALUE", str(pid))
            .replace("MODEL_BIN_VALUE", binary).replace("MODEL_PATH_VALUE", model))
