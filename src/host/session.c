#include <AEEStdErr.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsp_capabilities_utils.h"          // $HEXAGON_SDK_ROOT/utils/examples
#include "htp_ops.h"                         // QAIC auto-generated header for FastRPC

static remote_handle64 session_handle = -1;  // global session handle

remote_handle64 get_global_handle() {
  return session_handle;
}

int open_dsp_session(int domain_id, int unsigned_pd_enabled) {
  int   err        = AEE_SUCCESS;
  char *uri_domain = NULL;

  domain *my_domain = get_domain(domain_id);
  if (!my_domain) {
    err = AEE_EBADPARM;
    fprintf(stderr, "ERROR 0x%x: unable to get domain struct %d\n", err, domain_id);
    goto bail;
  }

  if (unsigned_pd_enabled) {
    if (&remote_session_control) {
      struct remote_rpc_control_unsigned_module ctrl;
      ctrl.domain = domain_id;
      ctrl.enable = 1;

      err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ctrl, sizeof(ctrl));
      if (err != AEE_SUCCESS) {
        fprintf(stderr, "ERROR 0x%x: remote_session_control failed\n", err);
        goto bail;
      }
    } else {
      err = AEE_EUNSUPPORTED;
      fprintf(stderr,
              "ERROR 0x%x: remote_session_control interface is not supported on "
              "this device\n",
              err);
      goto bail;
    }
  }

  int uri_domain_len = strlen(htp_ops_URI) + MAX_DOMAIN_URI_SIZE;
  uri_domain         = (char *) malloc(uri_domain_len);
  if (!uri_domain) {
    err = AEE_ENOMEMORY;
    fprintf(stderr, "unable to allocated memory for uri_domain of size: %d", uri_domain_len);
    goto bail;
  }

  err = snprintf(uri_domain, uri_domain_len, "%s%s", htp_ops_URI, my_domain->uri);
  if (err < 0) {
    fprintf(stderr, "ERROR 0x%x returned from snprintf\n", err);
    err = AEE_EFAILED;
    goto bail;
  }

  err = htp_ops_open(uri_domain, &session_handle);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "DSP session open failed: 0x%08x\n", (unsigned) err);
    goto bail;
  }

  // enable FastRPC QoS mode
  struct remote_rpc_control_latency lat_ctrl;
  lat_ctrl.enable = RPC_PM_QOS;
  lat_ctrl.latency = 50; // target latency: 50 us (not guaranteed)

  err = remote_handle64_control(session_handle, DSPRPC_CONTROL_LATENCY, &lat_ctrl, sizeof(lat_ctrl));
  if (err) {
    fprintf(stderr, "Enabling FastRPC QoS mode failed: 0x%08x\n", (unsigned) err);
    goto bail;
  }

bail:
  if (uri_domain) {
    free(uri_domain);
  }
  // return err;
  return err == AEE_SUCCESS ? 0 : -1;
}

void close_dsp_session() {
  htp_ops_close(session_handle);
}

void init_htp_backend() {
  htp_ops_init_backend(session_handle);
}

int create_htp_message_channel(int fd, unsigned int max_msg_size) {
  return htp_ops_create_channel(session_handle, fd, max_msg_size);
}
