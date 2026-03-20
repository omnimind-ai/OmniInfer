#include "ggml-htp.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-htp-impl.h"
#include "mse_checker.h"

// real backend initialization work is done here. ggml_backend_htp_init is only a wrapper
ggml_backend_htp_context::ggml_backend_htp_context() : mapper(3 * 1024UL * 1024 * 1024, true) {
    fprintf(stderr, "Initializing HTP backend... (You should see this once)\n");

    // rpcmem_init & rpcmem_deinit are actually not required on modern Hexagon processors
    rpcmem_init();

    // Try multiple possible paths for HTP ops library
    const char* htp_paths[] = {
        HTP_OPS_DL_PATH,
        "./libhtp_ops.so",
        nullptr
    };
    bool find_lib = false;
    fprintf(stderr, "Attempting to load HTP ops library...\n");
    for (int i = 0; htp_paths[i] != nullptr; i++) {
        fprintf(stderr, "Trying path: %s\n", htp_paths[i]);
        ops_dl_handle = dlopen(htp_paths[i], RTLD_LAZY | RTLD_LOCAL);
        if (ops_dl_handle != nullptr) {
            find_lib = true;
            break;
        } else {
            const char* error = dlerror();
        }
    }
    if (!find_lib) {
        fprintf(stderr, "❌ Failed to load HTP ops library\n");
        return;
    } else {
        fprintf(stderr, "✅ Successfully loaded HTP ops library\n");
    }

    if (ops_dl_handle != nullptr) {
        using open_session_fn_type = int(int, int);
        using init_htp_ops_fn_type = void();

        auto open_session = reinterpret_cast<open_session_fn_type *>(dlsym(ops_dl_handle, "open_dsp_session"));
        auto init_htp_ops = reinterpret_cast<init_htp_ops_fn_type *>(dlsym(ops_dl_handle, "init_htp_backend"));
        GGML_ASSERT(open_session && init_htp_ops);

        int err = open_session(CDSP_DOMAIN_ID, 1);
        if (err == 0) {
            init_htp_ops();

            if (init_message_channel() == 0) {
                ops_backend_initialized = true;
            }
        } else {
            fprintf(stderr, "Failed to open remote session on Hexagon NPU (0x%x)\n", err);
        }
    } else {
        fprintf(stderr, "Cannot load HTP ops backend library, all OPs will fallback to CPU implementation\n");
    }

    if (getenv("SKIP_HTP_OPS")) {
        skip_htp_ops = true;
    }
}

ggml_backend_htp_context::~ggml_backend_htp_context() {
    // 打印 MSE 统计结果
    // mse_stats_print_summary();
    
    delete[] work_data;

    if (ops_dl_handle) {
        if (ops_backend_initialized) {
            using close_session_fn = void();

            auto close_session = reinterpret_cast<close_session_fn *>(dlsym(ops_dl_handle, "close_dsp_session"));
            GGML_ASSERT(close_session);

            close_session();
            // release message channel
            fastrpc_munmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, MAX_MSG_SIZE);
            rpcmem_free(ops_msg_chan);
            ops_backend_initialized = false;
        }

        dlclose(ops_dl_handle);
    }

    rpcmem_deinit();
}

int ggml_backend_htp_context::init_message_channel() {
    using create_msg_channel_fn_type = int(int, unsigned int);

    auto create_msg_channel =
        reinterpret_cast<create_msg_channel_fn_type *>(dlsym(ops_dl_handle, "create_htp_message_channel"));
    if (!create_msg_channel) {
        return -1;
    }

    ops_msg_chan = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, MAX_MSG_SIZE);
    if (!ops_msg_chan) {
        return -1;
    }

    msg_chan_fd = rpcmem_to_fd(ops_msg_chan);
    if (msg_chan_fd < 0) {
        return -1;
    }

    int err = fastrpc_mmap(CDSP_DOMAIN_ID, msg_chan_fd, ops_msg_chan, 0, MAX_MSG_SIZE, FASTRPC_MAP_FD);
    if (err) {
        return -1;
    }

    return create_msg_channel(msg_chan_fd, MAX_MSG_SIZE);
}

// singleton
ggml_backend_htp_context * ggml_backend_htp_context::instance() {
    static std::unique_ptr<ggml_backend_htp_context> ctx_ptr;
    static std::once_flag                            ctx_once_flag;

    std::call_once(ctx_once_flag, [&] {
        auto * ctx = new ggml_backend_htp_context;
        ctx_ptr.reset(ctx);
    });
    return ctx_ptr.get();
}

// HTP backend buffer type (shared rpcmem)

static void * ggml_backend_htp_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_htp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    rpcmem_free(buffer->context);
}

static void ggml_backend_htp_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_htp_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_htp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_htp_buffer_i = {
    /* .free_buffer     = */ ggml_backend_htp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_htp_buffer_get_base,
    /* .init_tensor     = */ nullptr,  // no initialization required?
    /* .memset_tensor   = */ ggml_backend_htp_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_htp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_htp_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_htp_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_htp_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_htp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "RPCMEM";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_htp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, size);
    GGML_ASSERT(data);

    fprintf(stderr, "RPCMEM alloc size = %.5f MiB\n", size / 1024.0 / 1024.0);

    return ggml_backend_buffer_init(buft, ggml_backend_htp_buffer_i, data, size);
}

static size_t ggml_backend_htp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static bool ggml_backend_htp_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_htp_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    // TODO(hzx): change max size
    return 256 * size_t(1024 * 1024);

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_htp_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_htp_buffer_type = {
        /* .iface   = */ {
                          /* .get_name         = */ ggml_backend_htp_buffer_type_get_name,
                          /* .alloc_buffer     = */ ggml_backend_htp_buffer_type_alloc_buffer,
                          /* .get_alignment    = */ ggml_backend_htp_buffer_type_get_alignment,
                          /* .get_max_size     = */ ggml_backend_htp_buffer_type_get_max_size,
                          /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_htp_buffer_type_is_host,
                          },
        /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0),
        /* .context = */ nullptr,
    };

    return &ggml_backend_htp_buffer_type;
}

bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_htp_buffer_type_get_name;
}

// backend interface

static const char * ggml_backend_htp_get_name(ggml_backend_t backend) {
    return "MyHTP";

    GGML_UNUSED(backend);
}

static void ggml_backend_htp_free(ggml_backend_t backend) {
    // ggml_backend_htp_context * ctx = (ggml_backend_htp_context *) backend->context;
    // delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_htp_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    constexpr bool use_mock_cpu_backend = false;

    if (use_mock_cpu_backend) {
        // simple mock: use cpu backend
        static ggml_backend_t my_cpu_backend = nullptr;
        if (!my_cpu_backend) {
            auto * cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
            // my_cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            my_cpu_backend = ggml_backend_dev_init(cpu_dev, nullptr);
            GGML_ASSERT(my_cpu_backend);
        }

        return ggml_backend_graph_compute(my_cpu_backend, cgraph);
    }

    struct ggml_backend_htp_context * ctx = (struct ggml_backend_htp_context *) backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, ctx->n_threads, ctx->threadpool);

    if (ctx->work_size < cplan.work_size) {
        delete[] ctx->work_data;
        ctx->work_data = new uint8_t[cplan.work_size]{ 0 };
        if (!ctx->work_data) {
            ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
    }
    cplan.work_data = (uint8_t *) ctx->work_data;

    // cplan.abort_callback      = ctx->abort_callback;
    // cplan.abort_callback_data = ctx->abort_callback_data;

    return ggml_graph_compute_htp_hybrid(cgraph, &cplan);
}

static struct ggml_backend_i htp_backend_i = {
    /* .get_name                = */ ggml_backend_htp_get_name,
    /* .free                    = */ ggml_backend_htp_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_htp_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
};

static ggml_guid_t ggml_backend_htp_guid(void) {
    static ggml_guid guid = { 0x6b, 0x22, 0x31, 0xb2, 0xfb, 0x66, 0x46, 0xf6,
                              0x87, 0x3b, 0x7d, 0x8a, 0x13, 0xe5, 0x78, 0x13 };
    return &guid;
}

static ggml_backend_t ggml_backend_htp_init(void) {
    // ggml_backend_htp_context * ctx = new ggml_backend_htp_context;
    auto * ctx = ggml_backend_htp_context::instance();

    ggml_backend_t backend = new ggml_backend{
        /* .guid      = */ ggml_backend_htp_guid(),
        /* .interface = */ htp_backend_i,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_htp_reg(), 0),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_htp(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_htp_guid());
}

static const char * ggml_backend_htp_device_get_name(ggml_backend_dev_t dev) {
    return "HTP";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_htp_device_get_description(ggml_backend_dev_t dev) {
    return "Unknown Hexagon Processor";

    GGML_UNUSED(dev);
}

static void ggml_backend_htp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 4 * (1UL << 30);
    *total = 4 * (1UL << 30);

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_htp_device_get_type(ggml_backend_dev_t dev) {
    // TODO(hzx): use GGML_BACKEND_DEVICE_TYPE_GPU or GGML_BACKEND_DEVICE_TYPE_ACCEL?
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_htp_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_htp_device_get_name(dev);
    props->description = ggml_backend_htp_device_get_description(dev);
    props->type        = ggml_backend_htp_device_get_type(dev);
    ggml_backend_htp_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_htp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_htp_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_htp_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_htp_buffer_type();

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_backend_dev_supports_op(cpu_dev, op);

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_htp_buffer_type_get_name;

    GGML_UNUSED(dev);
}

static bool ggml_backend_htp_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_backend_dev_supports_op(cpu_dev, op);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_htp_device_i = {
    /* .get_name             = */ ggml_backend_htp_device_get_name,
    /* .get_description      = */ ggml_backend_htp_device_get_description,
    /* .get_memory           = */ ggml_backend_htp_device_get_memory,
    /* .get_type             = */ ggml_backend_htp_device_get_type,
    /* .get_props            = */ ggml_backend_htp_device_get_props,
    /* .init_backend         = */ ggml_backend_htp_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_htp_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_htp_device_supports_op,
    /* .supports_buft        = */ ggml_backend_htp_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// backend reg interface

static const char * ggml_backend_htp_reg_get_name(ggml_backend_reg_t reg) {
    return "MyHTP";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_htp_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_htp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    // TODO

    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_htp_device = {
        /* .iface   = */ ggml_backend_htp_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &ggml_backend_htp_device;
}

static const struct ggml_backend_reg_i ggml_backend_htp_reg_i = {
    /* .get_name         = */ ggml_backend_htp_reg_get_name,
    /* .get_device_count = */ ggml_backend_htp_reg_get_device_count,
    /* .get_device       = */ ggml_backend_htp_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_htp_reg(void) {
    static struct ggml_backend_reg ggml_backend_htp_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_htp_reg_i,
        /* .context     = */ nullptr,
    };
    return &ggml_backend_htp_reg;
}

// NOTE(hzx): GGML_BACKEND_DL is not set defaultly, so this should generate nothing
// TODO: investigate why NDK build emit warnings but local build do not
// The warning is something like "load_backend: failed to find ggml_backend_init in ./libggml-htp.so"
GGML_BACKEND_DL_IMPL(ggml_backend_htp_reg)
