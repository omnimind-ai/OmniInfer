#include "host/op_export.h"

#include "host/session.h"
#include "htp_ops.h"

int htp_ops_rpc_rms_norm_f32(int dst_fd, int dst_offset, int src_fd, int src_offset, int ne0, int ne1) {
  return htp_ops_rms_norm_f32(get_global_handle(), dst_fd, dst_offset, src_fd, src_offset, ne0, ne1);
}

int htp_ops_rpc_mat_mul_permuted_w16a32(int output_fd, int output_offset, int activation_fd, int activation_offset,
                                        int weight_fd, int weight_offset, int m, int k, int n) {
  return htp_ops_mat_mul_permuted_w16a32(get_global_handle(), output_fd, output_offset, activation_fd,
                                         activation_offset, weight_fd, weight_offset, m, k, n);
}
