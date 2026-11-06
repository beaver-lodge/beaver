/* An independent ABI fixture, not Batata's runtime implementation. */
#include "aot.h"
#include <stdlib.h>
#ifndef BTT_TEST_FIXTURE_ID
#define BTT_TEST_FIXTURE_ID "abi_fixture"
#endif
struct BttRuntime { uint64_t owner; unsigned roots,steps; int process,called; };
static int process_ok(BttRuntime *r,BttProcess p){return r && r->process && p.owner==r->owner && p.generation==UINT64_C(0xabcdef0123456789) && !p.slot && !p.reserved;}
static int value_ok(BttRuntime *r,BttValue v){return r && v.owner==r->owner && !v.process && v.slot<2 && v.generation==UINT64_C(0xfedcba9876543210)+v.slot && (r->roots&(1u<<v.slot));}
static BttValue root(BttRuntime *r,unsigned slot){r->roots|=1u<<slot;return (BttValue){r->owner,UINT64_C(0xfedcba9876543210)+slot,0,slot};}
int btt_query(uint64_t abi,BttQuery *out){if(abi!=BTT_ABI_ID || !out)return BTT_UNSUPPORTED;*out=(BttQuery){BTT_ABI_ID,1,1,BTT_TEST_FIXTURE_ID};return BTT_OK;}
int btt_create(uint64_t abi,BttRuntime **out){if(abi!=BTT_ABI_ID || !out)return BTT_UNSUPPORTED;*out=calloc(1,sizeof(BttRuntime));if(!*out)return BTT_FAILED;(*out)->owner=(uint64_t)(uintptr_t)*out;return BTT_OK;}
int btt_register(BttRuntime *r,uint64_t caps){return r && !caps ? BTT_OK : BTT_UNSUPPORTED;}
int btt_process(BttRuntime *r,BttEntry e,BttProcess *out){if(!r || e.program || e.index || !out || r->process)return BTT_INVALID;r->process=1;*out=(BttProcess){r->owner,UINT64_C(0xabcdef0123456789),0,0};return BTT_OK;}
int btt_value(BttRuntime *r,BttProcess p,uint32_t kind,const uint8_t *bytes,uint64_t length,const BttValue *children,uint32_t count,BttValue *out){if(!process_ok(r,p)||kind!=BTT_NIL||bytes||length||children||count||!out)return BTT_INVALID;*out=root(r,0);return BTT_OK;}
int btt_call(BttRuntime *r,BttProcess p,const BttValue *args,uint32_t count){if(!process_ok(r,p)||count!=1||!args||!value_ok(r,args[0]))return BTT_INVALID;r->called=1;return BTT_OK;}
int btt_step(BttRuntime *r,uint32_t budget,BttStep *out){if(!r||!budget||!out)return BTT_INVALID;r->steps++;*out=(BttStep){r->steps<3?BTT_OK:BTT_DONE,1,UINT64_C(0x100000000)+r->steps};return BTT_OK;}
int btt_result(BttRuntime *r,BttProcess p,BttValue *out,uint32_t *exception){if(!process_ok(r,p)||!out||!exception)return BTT_INVALID;if(!r->called||r->steps<3)return BTT_WAITING;*out=root(r,1);*exception=0;return BTT_OK;}
int btt_view(BttRuntime *r,BttValue v,BttView *out){if(!value_ok(r,v)||!out)return BTT_INVALID;*out=(BttView){BTT_NIL,0,NULL,0};return BTT_OK;}
int btt_release(BttRuntime *r,BttValue v){if(!value_ok(r,v))return BTT_INVALID;r->roots&=~(1u<<v.slot);return BTT_OK;}
int btt_cancel(BttRuntime *r,BttProcess p){return process_ok(r,p)?BTT_OK:BTT_INVALID;}
int btt_drain(BttRuntime *r,BttProcess p,uint32_t budget){if(!process_ok(r,p))return BTT_INVALID;if(!budget||r->roots)return BTT_BUSY;r->process=0;return BTT_OK;}
int btt_destroy(BttRuntime *r){if(!r)return BTT_INVALID;if(r->process)return BTT_BUSY;free(r);return BTT_OK;}
int btt_program(uint32_t i,const char **name,uint32_t *count){return BTT_UNSUPPORTED;}
int btt_entry(BttEntry e,const char **name,uint32_t *arity){return BTT_UNSUPPORTED;}
int btt_float(BttRuntime *r,BttProcess p,uint64_t q,uint64_t b,BttValue *v){return BTT_UNSUPPORTED;}
int btt_float_read(BttRuntime *r,BttValue v,uint64_t *bits){return BTT_UNSUPPORTED;}
int btt_atom(BttRuntime *r,BttProcess p,uint64_t q,const uint8_t *s,uint64_t n,BttValue *v){return BTT_UNSUPPORTED;}
int btt_callback(BttRuntime *r,BttProcess p,uint32_t id,uint32_t arity,BttValue *v){return BTT_UNSUPPORTED;}
int btt_export(BttRuntime *r,BttProcess p,BttEntry e,BttValue *v){return BTT_UNSUPPORTED;}
int btt_context_set(BttRuntime *r,BttProcess p,uint64_t q,BttValue v){return BTT_UNSUPPORTED;}
int btt_context_get(BttRuntime *r,BttProcess p,BttValue *v){return BTT_UNSUPPORTED;}
int btt_event(BttRuntime *r,BttEvent *e){return BTT_UNSUPPORTED;}
int btt_reply(BttRuntime *r,BttProcess p,uint64_t q,BttValue v,uint32_t exception){return BTT_UNSUPPORTED;}
int btt_child(BttRuntime *r,BttValue v,uint32_t i,BttValue *out){return BTT_UNSUPPORTED;}
int btt_key(BttRuntime *r,BttValue v,uint32_t i,BttValue *out){return BTT_UNSUPPORTED;}
/* Sentinel values below exercise ABI layouts, not import/copy semantics. */
int btt_closure_definition(uint32_t p,uint32_t n,BttCallable *out){
    if(p!=18 || n!=37 || !out)return BTT_INVALID;
    *out=(BttCallable){BTT_GUEST_CLOSURE,p,n,2,3,71,BTT_TEST_FIXTURE_ID,"closure"};return BTT_OK;
}
int btt_closure(BttRuntime * a0,BttProcess a1,uint64_t a2,uint32_t a3,uint32_t a4,const BttValue * a5,uint32_t a6,BttValue * a7){return BTT_UNSUPPORTED;}
int btt_opaque(BttRuntime * a0,BttProcess a1,uint64_t a2,uint32_t a3,uint32_t a4,BttValue * a5){return BTT_UNSUPPORTED;}
int btt_callable_info(BttRuntime * a0,BttValue a1,BttCallable * a2){return BTT_UNSUPPORTED;}
static int import_ok(BttRuntime *r,BttImport t){return r && t.owner==r->owner && t.process_generation==UINT64_C(0xabcdef0123456789) && t.generation==UINT64_C(0x8877665544332211) && t.process==19 && t.reserved==23;}
int btt_integer_begin(BttRuntime *r,BttProcess p,uint64_t request,uint64_t length,uint32_t sign,BttImport *out){
    if(!process_ok(r,p) || request!=UINT64_C(0x123456789abcdef0) || length!=UINT64_C(0x1020304050607080) || sign!=1 || !out)return BTT_INVALID;
    *out=(BttImport){r->owner,p.generation,UINT64_C(0x8877665544332211),19,23};return BTT_OK;
}
int btt_integer_write(BttRuntime *r,BttImport t,uint64_t offset,const uint8_t *bytes,uint32_t length){
    return import_ok(r,t) && offset==UINT64_C(0xaabbccddeeff0011) && bytes && length==3 && bytes[2]==255 ? BTT_OK : BTT_INVALID;
}
int btt_integer_commit(BttRuntime * a0,BttImport a1,BttValue * a2){return BTT_UNSUPPORTED;}
int btt_import_cancel(BttRuntime *r,BttImport t){return import_ok(r,t) ? BTT_OK : BTT_INVALID;}
int btt_binary_begin(BttRuntime *r,BttProcess p,uint64_t request,uint64_t length,BttImport *out){
    int status=btt_integer_begin(r,p,request,length,1,out);return status==BTT_OK ? BTT_FAILED : status;
}
int btt_binary_write(BttRuntime * a0,BttImport a1,uint64_t a2,const uint8_t * a3,uint32_t a4){return BTT_UNSUPPORTED;}
int btt_binary_commit(BttRuntime * a0,BttImport a1,BttValue * a2){return BTT_UNSUPPORTED;}
int btt_binary_origin(BttRuntime * a0,BttImport a1,uint32_t a2){return BTT_UNSUPPORTED;}
int btt_binary_owner(BttRuntime *r,BttValue v,BttValue *out,uint64_t *offset,uint32_t *external,uint32_t *origin){
    if(!value_ok(r,v) || !out || !offset || !external || !origin)return BTT_INVALID;
    *out=root(r,1);*offset=UINT64_C(0xf0e0d0c0b0a09080);*external=1;*origin=77;return BTT_OK;
}
int btt_binary_same_owner(BttRuntime * a0,BttValue a1,BttValue a2,uint32_t * a3){return BTT_UNSUPPORTED;}
int btt_term_begin(BttRuntime * a0,BttProcess a1,uint64_t a2,uint64_t a3,BttImport * a4){return BTT_UNSUPPORTED;}
int btt_term_write(BttRuntime * a0,BttImport a1,uint64_t a2,const uint8_t * a3,uint32_t a4){return BTT_UNSUPPORTED;}
int btt_term_bind(BttRuntime * a0,BttImport a1,uint32_t a2,BttValue a3){return BTT_UNSUPPORTED;}
int btt_term_start(BttRuntime * a0,BttImport a1){return BTT_UNSUPPORTED;}
int btt_term_commit(BttRuntime * a0,BttImport a1,BttValue * a2){return BTT_UNSUPPORTED;}
int btt_integer_read(BttRuntime *r,BttValue v,uint64_t offset,uint8_t *bytes,uint32_t capacity,uint32_t *written,uint64_t *length,uint32_t *negative){
    if(!value_ok(r,v) || offset!=UINT64_C(0x1122334455667788) || !bytes || capacity!=3 || !written || !length || !negative)return BTT_INVALID;
    bytes[0]=1;bytes[1]=128;bytes[2]=255;*written=3;*length=UINT64_C(0x1020304050607080);*negative=1;return BTT_OK;
}
int btt_copy_begin(BttRuntime *r,BttProcess p,BttValue v,BttCopy *out){
    if(!process_ok(r,p) || !value_ok(r,v) || !out)return BTT_INVALID;
    *out=(BttCopy){r->owner,UINT64_C(0x1122334455667788),29,31};return BTT_OK;
}
int btt_copy_step(BttRuntime *r,BttCopy job,uint32_t budget,BttValue *out,uint32_t *work){
    if(!r || job.owner!=r->owner || job.generation!=UINT64_C(0x1122334455667788) || job.process!=29 || job.reserved!=31 || budget!=13 || !out || !work)return BTT_INVALID;
    *out=root(r,1);*work=11;return BTT_DONE;
}
int btt_copy_cancel(BttRuntime * a0,BttCopy a1){return BTT_UNSUPPORTED;}
