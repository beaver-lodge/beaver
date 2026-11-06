/* Derive the JIT pointer ABI from the independent public ABI fixture. */
#include "aot.h"
#include <string.h>
#define BTT_TEST_VALUE(type) const type *
#define BTT_TEST_CALL(name, ...) int (*btt_##name)(__VA_ARGS__);
typedef struct {
#include "api.def"
} BttTestAPI;
#undef BTT_TEST_CALL
#undef BTT_TEST_VALUE
_Static_assert(sizeof(BttTestAPI)==49*sizeof(void*), "ABI test function table");
#define CHECK(x) do {if(!(x))return __LINE__;} while(0)
#define OK(x) CHECK((x)==BTT_OK)
#define TRY(x) do {int error=(x);if(error)return error;} while(0)
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
static int lifecycle(BttTestAPI *a,const char *identity) {
    BttQuery q;CHECK(a->btt_query(0,&q)==BTT_UNSUPPORTED);CHECK(a->btt_query(BTT_ABI_ID,&q)==BTT_OK && q.abi==BTT_ABI_ID && q.programs);
    CHECK(q.identity && strcmp(q.identity,identity)==0);
    CHECK(a->btt_query(UINT64_C(0x44414f540000000b),&q)==BTT_UNSUPPORTED);
    CHECK(a->btt_query(UINT64_C(0x425454000000000b),&q)==BTT_UNSUPPORTED);
    BttRuntime *r=NULL,*other=NULL;
    CHECK(a->btt_create(UINT64_C(0x44414f540000000b),&r)==BTT_UNSUPPORTED && !r);
    CHECK(a->btt_create(UINT64_C(0x425454000000000b),&r)==BTT_UNSUPPORTED && !r);
    CHECK(a->btt_create(BTT_ABI_ID,&r)==BTT_OK);CHECK(a->btt_create(BTT_ABI_ID,&other)==BTT_OK);
    CHECK(a->btt_register(r,0)==BTT_OK);BttEntry e={0,0};BttProcess p;
    CHECK(a->btt_process(r,NULL,&p)==BTT_INVALID);CHECK(a->btt_process(r,&e,&p)==BTT_OK);
    CHECK(a->btt_destroy(r)==BTT_BUSY);BttValue arg,out;uint32_t exception;
    CHECK(a->btt_value(r,&p,BTT_NIL,NULL,0,NULL,0,&arg)==BTT_OK);
    CHECK(a->btt_release(other,&arg)==BTT_INVALID);CHECK(a->btt_call(r,&p,&arg,1)==BTT_OK);
    BttStep progress;int status;unsigned turns=0;
    while((status=a->btt_result(r,&p,&out,&exception))==BTT_WAITING){CHECK(a->btt_step(r,1,&progress)==BTT_OK && progress.work<=1);CHECK(++turns<10000);}
    CHECK(status==BTT_OK && !exception);BttView v;CHECK(a->btt_view(r,&out,&v)==BTT_OK && v.kind==BTT_NIL);
    CHECK(a->btt_drain(r,&p,1)==BTT_BUSY);CHECK(a->btt_release(r,&arg)==BTT_OK);CHECK(a->btt_release(r,&out)==BTT_OK);
    CHECK(a->btt_release(r,&out)==BTT_INVALID);CHECK(a->btt_cancel(r,&p)==BTT_OK);turns=0;
    CHECK(a->btt_drain(r,&p,0)==BTT_BUSY);
    while((status=a->btt_drain(r,&p,1))==BTT_BUSY)CHECK(++turns<10000);
    CHECK(status==BTT_OK);CHECK(a->btt_cancel(r,&p)==BTT_INVALID);
    CHECK(a->btt_destroy(r)==BTT_OK);CHECK(a->btt_destroy(other)==BTT_OK);return 0;
}
