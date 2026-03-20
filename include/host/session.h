#pragma once

#include <remote.h>

int open_dsp_session(int domain_id, int unsigned_pd_enabled);
void close_dsp_session();

remote_handle64 get_global_handle();

void init_htp_backend();
int create_htp_message_channel(int fd, unsigned int max_msg_size);
