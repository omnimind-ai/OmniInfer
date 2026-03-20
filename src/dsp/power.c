#include "dsp/power.h"

#include <HAP_farf.h>
#include <HAP_power.h>

static int power_ctx;

// TODO(hzx): maybe we should set params according to SoC model
void power_setup() {
  int err;

  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;

  req.dcvs_v3.dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;

  req.dcvs_v3.set_latency = TRUE;
  req.dcvs_v3.latency     = 100;  // microseconds

  req.dcvs_v3.set_core_params           = TRUE;
  req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;

  req.dcvs_v3.set_bus_params           = TRUE;
  req.dcvs_v3.bus_params.min_corner    = HAP_DCVS_VCORNER_NOM;
  req.dcvs_v3.bus_params.max_corner    = HAP_DCVS_VCORNER_TURBO_L3;
  req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set DCVS v3 failed with return code 0x%x", err);
  }

  // power on HMX
  // NOTE(hzx): should we use v2 to set HMX clock frequency?
  memset(&req, 0, sizeof(req));
  req.type         = HAP_power_set_HMX;
  req.hmx.power_up = TRUE;

  err = HAP_power_set(&power_ctx, &req);
  if (err != AEE_SUCCESS) {
    FARF(ALWAYS, "HAP_power_set HMX failed with return code 0x%x", err);
  }
}

void power_reset() {
  HAP_power_request_t req;

  memset(&req, 0, sizeof(req));
  req.type         = HAP_power_set_HMX;
  req.hmx.power_up = FALSE;
  HAP_power_set(&power_ctx, &req);

  HAP_power_set_dcvs_v3_init(&req);
  HAP_power_set(&power_ctx, &req);
}
