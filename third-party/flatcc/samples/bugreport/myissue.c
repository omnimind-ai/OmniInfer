/* Minimal test with all headers generated into a single file. */
#include "build/myissue_generated.h"
#include "flatcc/support/hexdump.h"

int main(int argc, char *argv[])
{
    int ret;
    void *buf;
    size_t size;
    flatcc_builder_t builder, *B;

    (void)argc;
    (void)argv;

    B = &builder;
    flatcc_builder_init(B);

    Eclectic_FooBar_start_as_root(B);
    Eclectic_FooBar_say_create_str(B, "hello");
    Eclectic_FooBar_meal_add(B, Eclectic_Fruit_Orange);
    Eclectic_FooBar_height_add(B, -8000);
    Eclectic_FooBar_end_as_root(B);
    buf = flatcc_builder_get_direct_buffer(B, &size);
#if defined(PROVOKE_ERROR) || 0
    /* Provoke error for testing. */
    ((char*)buf)[0] = 42;
#endif
    ret = Eclectic_FooBar_verify_as_root(buf, size);
    if (ret) {
        hexdump("Eclectic.FooBar buffer for myissue", buf, size, stdout);
        printf("could not verify Electic.FooBar table, got %s\n", flatcc_verify_error_string(ret));
    }
    flatcc_builder_clear(B);
    return ret;
}
