// tests the optional event_completed slot of ggml_backend_event_completed

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#define CHECK(cond)                                                                                                  \
    do {                                                                                                             \
        if (!(cond)) {                                                                                               \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                                 \
            exit(1);                                                                                                 \
        }                                                                                                            \
    } while (0)

// a device without any optional interface slots, including event_completed
static ggml_backend_device g_dev_no_completed = {};

// a backend that does not implement the slot reports capability false
static void test_capability_false(void) {
    ggml_backend_event evt = { &g_dev_no_completed, nullptr };

    CHECK(ggml_backend_event_completed(&evt) == false);
}

// the contract over a recorded event: once the event has completed, the test reports true; polling waits
// for enqueued work without ever synchronizing the backend. skipped for devices without events, tolerated
// for devices that keep capability false
static void test_event_completed(ggml_backend_dev_t dev) {
    const char * name = ggml_backend_dev_name(dev);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "%s: skipping, failed to init the backend\n", name);
        return;
    }

    ggml_backend_event_t evt = ggml_backend_event_new(dev);
    if (evt == nullptr) {
        fprintf(stderr, "%s: skipping, the device has no events\n", name);
        ggml_backend_free(backend);
        return;
    }

    const size_t nbytes = 64ull*1024*1024;

    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    CHECK(ctx.get() != nullptr);

    ggml_tensor * t = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nbytes/4);
    CHECK(t != nullptr);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(dev)));
    CHECK(buf.get() != nullptr);

    std::vector<uint8_t> src(nbytes, 0xa5);
    ggml_backend_tensor_set(t, src.data(), 0, nbytes);

    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_buffer_t    buf_host    = buft_host ? ggml_backend_buft_alloc_buffer(buft_host, nbytes) : nullptr;
    std::vector<uint8_t>     staging;
    uint8_t *                host = buf_host ? (uint8_t *) ggml_backend_buffer_get_base(buf_host) : nullptr;
    if (host == nullptr) {
        staging.resize(nbytes);
        host = staging.data();
    }

    ggml_backend_tensor_get_async(backend, t, host, 0, nbytes);
    ggml_backend_event_record(evt, backend);
    ggml_backend_event_synchronize(evt);

    // after synchronize the event is complete, so a false test means the slot is not implemented
    if (!ggml_backend_event_completed(evt)) {
        fprintf(stderr, "%s: skipping the completed-event checks, capability false\n", name);
        ggml_backend_event_free(evt);
        if (buf_host) {
            ggml_backend_buffer_free(buf_host);
        }
        ggml_backend_free(backend);
        return;
    }

    // a second read keeps the event in flight, the poll must observe it complete
    ggml_backend_tensor_get_async(backend, t, host, 0, nbytes);
    ggml_backend_event_record(evt, backend);

    const double t0 = (double) std::clock() / CLOCKS_PER_SEC;
    while (!ggml_backend_event_completed(evt)) {
        const double dt = (double) std::clock() / CLOCKS_PER_SEC - t0;
        CHECK(dt < 60.0); // liveness: the wait must end on stream work alone
    }
    CHECK(ggml_backend_event_completed(evt));

    // the host copy landed
    CHECK(host[0] == 0xa5 && host[nbytes - 1] == 0xa5);

    ggml_backend_event_free(evt);
    if (buf_host) {
        ggml_backend_buffer_free(buf_host);
    }
    ggml_backend_free(backend);
}

int main(void) {
    test_capability_false();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            test_event_completed(dev);
        }
    }

    printf("test-event-completed: OK\n");
    return 0;
}
