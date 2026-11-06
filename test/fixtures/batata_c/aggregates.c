/* Independent sentinel ABI oracle. It makes no claims about Batata semantics. */
#include "bridge.h"
EXPORT int beaver_batata_aggregates(void **functions,const char *identity) {
    BttTestAPI a;memcpy(&a,functions,sizeof(a));TRY(lifecycle(&a,identity));
    BttCallable info;OK(a.btt_closure_definition(18,37,&info));
    CHECK(info.kind==BTT_GUEST_CLOSURE && info.program==18 && info.node==37 && info.arity==2 && info.captures==3 && info.host_id==71);
    CHECK(!strcmp(info.program_identity,identity) && !strcmp(info.identity,"closure"));
    BttRuntime *r=NULL;BttEntry entry={0,0};BttProcess p;BttValue v,out;
    OK(a.btt_create(BTT_ABI_ID,&r));OK(a.btt_process(r,&entry,&p));
    OK(a.btt_value(r,&p,BTT_NIL,NULL,0,NULL,0,&v));
    BttImport token;
    OK(a.btt_integer_begin(r,&p,UINT64_C(0x123456789abcdef0),UINT64_C(0x1020304050607080),1,&token));
    CHECK(token.owner==p.owner && token.process_generation==p.generation && token.generation==UINT64_C(0x8877665544332211) && token.process==19 && token.reserved==23);
    uint8_t bytes[]={1,128,255};
    OK(a.btt_integer_write(r,&token,UINT64_C(0xaabbccddeeff0011),bytes,3));OK(a.btt_import_cancel(r,&token));
    memset(&token,0,sizeof(token));
    CHECK(a.btt_binary_begin(r,&p,UINT64_C(0x123456789abcdef0),UINT64_C(0x1020304050607080),&token)==BTT_FAILED);
    CHECK(token.owner==p.owner && token.generation==UINT64_C(0x8877665544332211));OK(a.btt_import_cancel(r,&token));
    uint32_t count,negative,external,origin;uint64_t length,offset;
    memset(bytes,0,sizeof(bytes));
    OK(a.btt_integer_read(r,&v,UINT64_C(0x1122334455667788),bytes,3,&count,&length,&negative));
    CHECK(bytes[0]==1 && bytes[1]==128 && bytes[2]==255 && count==3 && length==UINT64_C(0x1020304050607080) && negative==1);
    OK(a.btt_binary_owner(r,&v,&out,&offset,&external,&origin));
    CHECK(offset==UINT64_C(0xf0e0d0c0b0a09080) && external==1 && origin==77 && out.owner==p.owner && out.slot==1);
    OK(a.btt_release(r,&out));
    BttCopy job;OK(a.btt_copy_begin(r,&p,&v,&job));
    CHECK(job.owner==p.owner && job.generation==UINT64_C(0x1122334455667788) && job.process==29 && job.reserved==31);
    CHECK(a.btt_copy_step(r,&job,13,&out,&count)==BTT_DONE && count==11 && out.slot==1 && out.generation==UINT64_C(0xfedcba9876543211));
    OK(a.btt_release(r,&out));OK(a.btt_release(r,&v));OK(a.btt_cancel(r,&p));OK(a.btt_drain(r,&p,1));OK(a.btt_destroy(r));
    return 0;
}
