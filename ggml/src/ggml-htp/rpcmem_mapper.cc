#include "rpcmem_mapper.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"

void RpcMemMapper::validate(const ggml_tensor * dst) {
    std::vector<ggml_backend_buffer *> buffers;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        if (ggml_backend_buft_is_rpcmem(buf->buft)) {
            buffers.push_back(buf);
        }
    };

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src) {
            add_buffer(src->buffer);
        }
    }
    add_buffer(dst->buffer);

    size_t required_size = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        if (buf_mapping.count(buf_base)) {
            // buffer already mapped, we put it at the front of the LRU access list
            accessed_bufs.erase(buf_iters.at(buf_base));
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
        } else {
            required_size += buf->size;
        }
    }

    GGML_ASSERT(required_size <= max_active_map_size);
    while (active_map_size + required_size > max_active_map_size) {
        // remove least recent used mapping
        void * buf_base     = accessed_bufs.back();
        auto [fd, buf_size] = buf_mapping.at(buf_base);

        if (defer_unmap) {
            // We assume slightly exceeding the planned max_active_map_size is acceptable
            pending_unmap_reqs.emplace_back(fd, buf_base, buf_size);
        } else {
            // fprintf(stderr, "rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
            //          buf_size / 1048576.0, fd);
            int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
            if (err) {
                fprintf(stderr, "fastrpc_munmap failed with return code: %x\n", err);
            }
            ++n_map_ops;
        }

        accessed_bufs.pop_back();
        buf_iters.erase(buf_base);
        buf_mapping.erase(buf_base);
        active_map_size -= buf_size;
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (buf_mapping.count(buf_base)) {
            continue;
        }

        int fd = rpcmem_to_fd(buf_base);
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n", fd, buf_base, dst->name,
                       n_map_ops);
        }

        auto it = std::find_if(pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
                               [fd](const auto & v) { return std::get<0>(v) == fd; });
        if (it == pending_unmap_reqs.end()) {
            int retry = 0;

            // TODO(hzx): fix this and remove reference code
            // fastrpc_munmap does not release DSP's vm mapping immediately
again:
            int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
            if (err) {
                dump_state();

                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    auto src = dst->src[i];
                    if (!src) {
                        continue;
                    }
                    fprintf(stderr, "  src index %d name %s", i, src->name);
                    if (ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
                        auto base = ggml_backend_buffer_get_base(src->buffer);
                        fprintf(stderr, " buf %p\n", base);
                    } else {
                        fprintf(stderr, " (non rpcmem)\n");
                    }
                }

                fprintf(stderr, "dst name: %s, op: %s, n buffers to map in this step: %ld, active size: %.2f MiB\n",
                        dst->name, ggml_op_name(dst->op), buffers.size(), active_map_size / 1048576.0);

                /*
                if (++retry < 3) {
                    // try: unmap all other buffers
                    std::unordered_set<void *> buf_ptrs;
                    for (auto b : buffers) {
                        buf_ptrs.insert(ggml_backend_buffer_get_base(b));
                    }

                    for (auto it = buf_mapping.begin(); it != buf_mapping.end();) {
                        auto ptr = it->first;
                        if (buf_ptrs.count(ptr)) {
                            ++it;
                            continue;  // skip needed buffers
                        }
                        auto [fd, len] = it->second;
                        auto lru_iter  = buf_iters.at(ptr);
                        accessed_bufs.erase(lru_iter);
                        buf_iters.erase(ptr);
                        it = buf_mapping.erase(it);

                        int e = fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, len);
                        fprintf(stderr, "try unmap fd %d addr %p len %ld ret %d\n", fd, ptr, len, e);

                        rpcmem_free(ptr);
                        fprintf(stderr, "danger operation: free %p\n", ptr);
                    }

                    goto again;
                }*/

                GGML_ABORT(
                    "fastrpc_mmap failed with return code: 0x%x fd: %d buf_base: %p buf_size: %ld buf usage: %d\n", err,
                    fd, buf_base, buf_size, buf->usage);
            }

            ++n_map_ops;
        } else {
            pending_unmap_reqs.erase(it);
        }

        // fprintf(stderr, "rpcmem_mapper: creating memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
        //         buf_size / 1048576.0, fd);

        accessed_bufs.push_front(buf_base);
        buf_iters[buf_base]   = accessed_bufs.begin();
        buf_mapping[buf_base] = { fd, buf_size };
        active_map_size += buf_size;
    }
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor * tensor) const {
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(tensor->buffer->buft));

    void * buf_base = ggml_backend_buffer_get_base(tensor->buffer);
    auto [fd, _]    = buf_mapping.at(buf_base);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    return { fd, offset };
}

void RpcMemMapper::unmap_all_pending_buffers() {
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        auto [fd, buf_base, buf_size] = *it;

        // fprintf(stderr, "rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
        //         buf_size / 1048576.0, fd);
        int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
        if (err) {
            fprintf(stderr, "fastrpc_munmap failed with return code: 0x%x\n", err);
        }
        ++n_map_ops;
        it = pending_unmap_reqs.erase(it);
    }
}

void RpcMemMapper::dump_state() const {
    fprintf(stderr, "total %d fastrpc_mmap + fastrpc_munmap ops\n", n_map_ops);

    if (!buf_mapping.empty()) {
        fprintf(stderr, "active mappings:\n");
    }
    for (const auto & [addr, pair] : buf_mapping) {
        fprintf(stderr, "    addr %p -> fd %d, size %.2f MiB\n", addr, pair.first, pair.second / 1048576.0);
    }

    if (!pending_unmap_reqs.empty()) {
        fprintf(stderr, "pending unmap requests:\n");
    }
    for (const auto & [fd, addr, size] : pending_unmap_reqs) {
        fprintf(stderr, "    fd %d, addr %p, size %.2f MiB\n", fd, addr, size / 1048576.0);
    }
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    ggml_backend_htp_context::instance()->mapper.validate(dst);
    return 0;
}
}
