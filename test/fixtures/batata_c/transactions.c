/* Black-box public ABI tests. All calls go through MLIR-generated functions. */
#include "bridge.h"
static int tick(BttTestAPI *a,BttRuntime *r) {
    BttStep step;OK(a->btt_step(r,1,&step));CHECK(step.work<=1);return 0;
}
static int finish(BttTestAPI *a,BttRuntime *r,BttProcess *p) {
    OK(a->btt_cancel(r,p));int status;unsigned turns=0;
    while((status=a->btt_drain(r,p,1))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && a->btt_cancel(r,p)==BTT_INVALID);return 0;
}
static int cancel_import(BttTestAPI *a,BttRuntime *r,BttImport *token) {
    int status;unsigned turns=0;
    while((status=a->btt_import_cancel(r,token))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && a->btt_import_cancel(r,token)==BTT_INVALID);return 0;
}
static int number(BttTestAPI *a,BttRuntime *r,BttProcess *p,const uint8_t *bytes,uint64_t size,unsigned negative,BttValue *out) {
    BttImport token={0};OK(a->btt_integer_begin(r,p,0,size,negative,&token));
    for(uint64_t at=0;at<size;){uint32_t n=size-at>137?137:(uint32_t)(size-at);OK(a->btt_integer_write(r,&token,at,bytes+at,n));at+=n;}
    OK(a->btt_integer_commit(r,&token,out));CHECK(a->btt_import_cancel(r,&token)==BTT_INVALID);return 0;
}
static int number_check(BttTestAPI *a,BttRuntime *r,BttValue *value,const uint8_t *expected,uint64_t size,unsigned sign) {
    BttView v;OK(a->btt_view(r,value,&v));CHECK(v.kind==BTT_INTEGER && !v.bytes && v.length==size);
    uint8_t bytes[256];uint64_t total,at=0;uint32_t written,negative;
    do {
        OK(a->btt_integer_read(r,value,at,bytes,256,&written,&total,&negative));
        CHECK(total==size && negative==sign && written<=256 && written<=size-at);
        CHECK(!memcmp(bytes,expected+at,written));at+=written;
    } while(at<size);
    OK(a->btt_integer_read(r,value,size,NULL,0,&written,&total,&negative));CHECK(!written);
    CHECK(a->btt_integer_read(r,value,size+1,bytes,1,&written,&total,&negative)==BTT_INVALID);
    return 0;
}
static int decode(BttTestAPI *a,BttRuntime *r,BttProcess *p,const uint8_t *bytes,uint64_t size,BttValue *out) {
    BttImport token={0};OK(a->btt_term_begin(r,p,0,size,&token));
    for(uint64_t at=0;at<size;){uint32_t n=size-at>193?193:(uint32_t)(size-at);OK(a->btt_term_write(r,&token,at,bytes+at,n));at+=n;}
    int status;unsigned turns=0;
    while((status=a->btt_term_start(r,&token))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK);
    while((status=a->btt_term_commit(r,&token,out))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && a->btt_term_start(r,&token)==BTT_INVALID);return 0;
}
static void u32(uint8_t *wire,size_t *n,uint32_t value) {
    for(int shift=24;shift>=0;shift-=8)wire[(*n)++]=(uint8_t)(value>>shift);
}
static size_t compound(uint8_t *wire,const uint8_t *bytes) {
    size_t n=0;wire[n++]=131;wire[n++]=104;wire[n++]=7;
    wire[n++]=119;wire[n++]=3;wire[n++]='a';wire[n++]=0;wire[n++]='b';
    wire[n++]=111;u32(wire,&n,513);wire[n++]=1;memcpy(wire+n,bytes,513);n+=513;
    wire[n++]=70;wire[n++]=128;memset(wire+n,0,7);n+=7;
    wire[n++]=109;u32(wire,&n,513);memcpy(wire+n,bytes,513);n+=513;
    wire[n++]=108;u32(wire,&n,2);wire[n++]=97;wire[n++]=42;wire[n++]=97;wire[n++]=43;wire[n++]=106;
    wire[n++]=116;u32(wire,&n,1);wire[n++]=97;wire[n++]=7;wire[n++]=119;wire[n++]=2;wire[n++]='o';wire[n++]='k';
    wire[n++]=108;u32(wire,&n,1);wire[n++]=97;wire[n++]=9;wire[n++]=119;wire[n++]=4;memcpy(wire+n,"tail",4);n+=4;
    return n;
}
static int compound_check(BttTestAPI *a,BttRuntime *r,BttValue *root,const uint8_t *bytes) {
    BttView view;OK(a->btt_view(r,root,&view));CHECK(view.kind==BTT_TUPLE && view.count==7);
    uint32_t kinds[]={BTT_ATOM,BTT_INTEGER,BTT_FLOAT,BTT_BINARY,BTT_CONS,BTT_MAP,BTT_CONS};
    for(unsigned i=0;i<7;i++) {
        BttValue child;OK(a->btt_child(r,root,i,&child));OK(a->btt_view(r,&child,&view));CHECK(view.kind==kinds[i]);
        if(i==0)CHECK(view.length==3 && !memcmp(view.bytes,"a\0b",3));
        if(i==1)TRY(number_check(a,r,&child,bytes,513,1));
        if(i==2){uint64_t bits;OK(a->btt_float_read(r,&child,&bits));CHECK(bits==UINT64_C(0x8000000000000000));}
        if(i==3)CHECK(view.length==513 && !memcmp(view.bytes,bytes,513));
        if(i==5){BttValue key;OK(a->btt_key(r,&child,0,&key));TRY(number_check(a,r,&key,(uint8_t[]){7},1,0));OK(a->btt_release(r,&key));}
        if(i==6){BttValue tail;OK(a->btt_child(r,&child,1,&tail));OK(a->btt_view(r,&tail,&view));CHECK(view.kind==BTT_ATOM && view.length==4 && !memcmp(view.bytes,"tail",4));OK(a->btt_release(r,&tail));}
        OK(a->btt_release(r,&child));
    }
    return 0;
}
static int imports_and_copy(BttTestAPI *a,BttRuntime *r,BttRuntime *foreign,BttEntry *entry) {
    BttProcess from,to;OK(a->btt_process(r,entry,&from));OK(a->btt_process(r,entry,&to));
    uint8_t bytes[1025];for(unsigned i=0;i<sizeof(bytes);i++)bytes[i]=(uint8_t)(i*37+1);
    BttImport token={0};BttValue integer={0};OK(a->btt_integer_begin(r,&from,0,513,1,&token));
    CHECK(a->btt_integer_write(foreign,&token,0,bytes,1)==BTT_INVALID);
    CHECK(a->btt_integer_write(r,&token,1,bytes,1)==BTT_INVALID);
    CHECK(a->btt_integer_write(r,&token,0,bytes,257)==BTT_INVALID);
    CHECK(a->btt_integer_commit(r,&token,&integer)==BTT_BUSY && !integer.owner);
    for(unsigned at=0;at<513;at+=256)OK(a->btt_integer_write(r,&token,at,bytes+at,513-at>256?256:513-at));
    BttImport stale=token;stale.process_generation++;CHECK(a->btt_integer_commit(r,&stale,&integer)==BTT_INVALID);
    OK(a->btt_integer_commit(r,&token,&integer));TRY(number_check(a,r,&integer,bytes,513,1));OK(a->btt_release(r,&integer));
    CHECK(a->btt_integer_write(r,&token,0,bytes,1)==BTT_INVALID);
    OK(a->btt_integer_begin(r,&from,0,1,0,&token));OK(a->btt_integer_write(r,&token,0,(uint8_t[]){0},1));
    CHECK(a->btt_integer_commit(r,&token,&integer)==BTT_INVALID);TRY(cancel_import(a,r,&token));
    TRY(number(a,r,&from,bytes,0,0,&integer));TRY(number_check(a,r,&integer,bytes,0,0));OK(a->btt_release(r,&integer));

    BttValue binary={0},owner;OK(a->btt_binary_begin(r,&from,0,sizeof(bytes),&token));
    CHECK(a->btt_integer_write(r,&token,0,bytes,1)==BTT_INVALID);OK(a->btt_binary_origin(r,&token,77));
    for(unsigned at=0;at<sizeof(bytes);at+=256)OK(a->btt_binary_write(r,&token,at,bytes+at,sizeof(bytes)-at>256?256:sizeof(bytes)-at));
    OK(a->btt_binary_commit(r,&token,&binary));BttView view;OK(a->btt_view(r,&binary,&view));CHECK(view.length==sizeof(bytes) && !memcmp(view.bytes,bytes,sizeof(bytes)));
    uint64_t offset;uint32_t external,origin,same;OK(a->btt_binary_owner(r,&binary,&owner,&offset,&external,&origin));
    CHECK(!offset && external && origin==77);OK(a->btt_binary_same_owner(r,&binary,&owner,&same));CHECK(same);OK(a->btt_release(r,&owner));

    BttCopy job;BttValue copied={0};uint32_t work;int status;unsigned turns=0;
    CHECK(a->btt_copy_begin(r,&from,&binary,&job)==BTT_INVALID);
    OK(a->btt_copy_begin(r,&to,&binary,&job));CHECK(a->btt_cancel(r,&to)==BTT_BUSY);
    OK(a->btt_copy_step(r,&job,0,&copied,&work));CHECK(!work && !copied.owner);
    OK(a->btt_copy_step(r,&job,1,&copied,&work));CHECK(work<=1);OK(a->btt_copy_cancel(r,&job));
    while((status=a->btt_copy_step(r,&job,1,&copied,&work))==BTT_OK)CHECK(work<=1 && ++turns<10000);
    CHECK(status==BTT_CANCELLED && !copied.owner && a->btt_copy_cancel(r,&job)==BTT_INVALID);OK(a->btt_release(r,&binary));

    uint8_t wire[2048];size_t size=compound(wire,bytes);BttValue root;
    TRY(decode(a,r,&from,wire,size,&root));TRY(compound_check(a,r,&root,bytes));
    OK(a->btt_copy_begin(r,&to,&root,&job));OK(a->btt_release(r,&root));OK(a->btt_cancel(r,&from));
    CHECK(a->btt_drain(r,&from,1)==BTT_BUSY);turns=0;
    while((status=a->btt_copy_step(r,&job,1,&copied,&work))==BTT_OK)CHECK(work<=1 && ++turns<100000);
    CHECK(status==BTT_DONE && work<=1 && copied.process==to.slot);
    CHECK(a->btt_copy_step(r,&job,1,&root,&work)==BTT_INVALID);TRY(finish(a,r,&from));
    TRY(compound_check(a,r,&copied,bytes));OK(a->btt_call(r,&to,&copied,1));
    BttValue result;uint32_t exception;turns=0;
    while((status=a->btt_result(r,&to,&result,&exception))==BTT_WAITING){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && !exception);TRY(compound_check(a,r,&result,bytes));
    OK(a->btt_release(r,&copied));OK(a->btt_release(r,&result));TRY(finish(a,r,&to));return 0;
}

static int rollback_and_references(BttTestAPI *a,BttRuntime *r,BttEntry *entry) {
    BttProcess p,other;OK(a->btt_process(r,entry,&p));OK(a->btt_process(r,entry,&other));
    BttImport token={0};BttValue out={0};uint8_t bad[]={131,255};OK(a->btt_term_begin(r,&p,0,sizeof(bad),&token));
    OK(a->btt_term_write(r,&token,0,bad,sizeof(bad)));OK(a->btt_term_start(r,&token));int status;unsigned turns=0;
    while((status=a->btt_term_commit(r,&token,&out))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<10000);}
    CHECK(status==BTT_INVALID && !out.owner);TRY(cancel_import(a,r,&token));
    uint8_t bytes[513],wire[2048];memset(bytes,1,sizeof(bytes));size_t size=compound(wire,bytes);
    OK(a->btt_term_begin(r,&p,0,size,&token));
    for(unsigned at=0;at<size;at+=256)OK(a->btt_term_write(r,&token,at,wire+at,size-at>256?256:size-at));
    OK(a->btt_term_start(r,&token));TRY(tick(a,r));TRY(cancel_import(a,r,&token));
    BttValue fun,foreign;OK(a->btt_export(r,&p,entry,&fun));OK(a->btt_export(r,&other,entry,&foreign));
    uint8_t reference[]={131,104,2,246,0,0,0,0,246,0,0,0,0};
    OK(a->btt_term_begin(r,&p,0,sizeof(reference),&token));OK(a->btt_term_write(r,&token,0,reference,sizeof(reference)));
    CHECK(a->btt_term_bind(r,&token,0,&foreign)==BTT_INVALID);CHECK(a->btt_term_bind(r,&token,128,&fun)==BTT_INVALID);
    OK(a->btt_term_bind(r,&token,0,&fun));CHECK(a->btt_term_bind(r,&token,0,&fun)==BTT_INVALID);OK(a->btt_release(r,&fun));
    CHECK(a->btt_term_bind(r,&token,1,&fun)==BTT_INVALID);OK(a->btt_term_start(r,&token));turns=0;
    while((status=a->btt_term_commit(r,&token,&out))==BTT_BUSY){TRY(tick(a,r));CHECK(++turns<10000);}
    CHECK(status==BTT_OK);BttValue child;BttCallable info;OK(a->btt_child(r,&out,0,&child));OK(a->btt_callable_info(r,&child,&info));CHECK(info.kind==BTT_GUEST_EXPORT && info.arity==1);
    OK(a->btt_release(r,&child));OK(a->btt_release(r,&out));OK(a->btt_release(r,&foreign));
    TRY(finish(a,r,&p));TRY(finish(a,r,&other));return 0;
}

static int await_result(BttTestAPI *a,BttRuntime *r,BttProcess *p,BttValue *out) {
    int status;unsigned turns=0;uint32_t exception;
    while((status=a->btt_result(r,p,out,&exception))==BTT_WAITING){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && !exception);return 0;
}

static int callables_and_events(BttTestAPI *a,BttRuntime *r) {
    const char *program,*identity;uint32_t count,arity;BttEntry entry={0,0};
    OK(a->btt_program(0,&program,&count));CHECK(program && count==3);
    for(entry.index=0;entry.index<count;entry.index++) {
        OK(a->btt_entry(&entry,&identity,&arity));if(arity==2)break;
    }
    CHECK(entry.index<count && strstr(identity,"invoke/2"));
    BttCallable definition,info;uint32_t node;int status;
    for(node=0;node<10000;node++) {
        status=a->btt_closure_definition(0,node,&definition);
        if(status==BTT_OK)break;
        CHECK(status==BTT_UNSUPPORTED);
    }
    CHECK(node<10000 && definition.kind==BTT_GUEST_CLOSURE && definition.arity==1 && definition.captures==1);
    CHECK(!strcmp(definition.program_identity,program) && definition.identity);
    BttProcess p;OK(a->btt_process(r,&entry,&p));BttValue capture,fun,arg,out;
    TRY(number(a,r,&p,(uint8_t[]){40},1,0,&capture));
    CHECK(a->btt_closure(r,&p,0,0,node,NULL,0,&fun)==BTT_INVALID);
    OK(a->btt_closure(r,&p,0,0,node,&capture,1,&fun));OK(a->btt_release(r,&capture));
    OK(a->btt_callable_info(r,&fun,&info));CHECK(info.kind==BTT_GUEST_CLOSURE && info.node==node && info.captures==1);
    TRY(number(a,r,&p,(uint8_t[]){2},1,0,&arg));
    BttValue args[]={fun,arg};OK(a->btt_call(r,&p,args,2));TRY(await_result(a,r,&p,&out));
    TRY(number_check(a,r,&out,(uint8_t[]){42},1,0));
    OK(a->btt_release(r,&fun));OK(a->btt_release(r,&arg));OK(a->btt_release(r,&out));TRY(finish(a,r,&p));

    OK(a->btt_process(r,&entry,&p));
    OK(a->btt_callback(r,&p,91,1,&fun));OK(a->btt_callable_info(r,&fun,&info));
    CHECK(info.kind==BTT_HOST_CALLABLE && info.arity==1 && info.host_id==91);
    TRY(number(a,r,&p,(uint8_t[]){7},1,0,&arg));OK(a->btt_context_set(r,&p,0,&arg));
    args[0]=fun;args[1]=arg;OK(a->btt_call(r,&p,args,2));
    BttEvent event;unsigned turns=0;
    while((status=a->btt_event(r,&event))==BTT_WAITING){TRY(tick(a,r));CHECK(++turns<100000);}
    CHECK(status==BTT_OK && event.kind==BTT_EVENT_CALLBACK && event.callback==91 && event.arity==1 && event.request);
    CHECK(event.process.generation==p.generation && event.process.slot==p.slot);
    TRY(number_check(a,r,&event.arguments[0],(uint8_t[]){7},1,0));
    OK(a->btt_context_get(r,&p,&out));TRY(number_check(a,r,&out,(uint8_t[]){7},1,0));OK(a->btt_release(r,&out));
    uint64_t bits=UINT64_C(0x8000000000000000),read;
    CHECK(a->btt_float(r,&p,0,bits,&out)==BTT_BUSY);
    OK(a->btt_float(r,&p,event.request,bits,&out));OK(a->btt_context_set(r,&p,event.request,&out));
    CHECK(a->btt_reply(r,&p,event.request+1,&out,0)==BTT_INVALID);
    OK(a->btt_reply(r,&p,event.request,&out,0));OK(a->btt_release(r,&out));
    OK(a->btt_release(r,&event.arguments[0]));OK(a->btt_release(r,&arg));OK(a->btt_release(r,&fun));
    TRY(await_result(a,r,&p,&out));OK(a->btt_float_read(r,&out,&read));CHECK(read==bits);OK(a->btt_release(r,&out));
    OK(a->btt_context_get(r,&p,&out));OK(a->btt_float_read(r,&out,&read));CHECK(read==bits);OK(a->btt_release(r,&out));
    TRY(finish(a,r,&p));return 0;
}

static int constructors_and_capabilities(BttTestAPI *a,BttRuntime *r,BttEntry *entry) {
    BttProcess p,to;OK(a->btt_process(r,entry,&p));OK(a->btt_process(r,entry,&to));
    BttValue value;uint64_t read;
    uint64_t floats[]={0,UINT64_C(0x8000000000000000),1,UINT64_C(0x7fefffffffffffff)};
    for(unsigned i=0;i<sizeof(floats)/sizeof(*floats);i++) {
        OK(a->btt_float(r,&p,0,floats[i],&value));OK(a->btt_float_read(r,&value,&read));
        CHECK(read==floats[i]);OK(a->btt_release(r,&value));
    }
    CHECK(a->btt_float(r,&p,0,UINT64_C(0x7ff0000000000000),&value)==BTT_UNSUPPORTED);
    CHECK(a->btt_float(r,&p,0,UINT64_C(0x7ff8000000000000),&value)==BTT_UNSUPPORTED);
    BttValue atom,nil,cons,tuple,map;BttView view;
    OK(a->btt_atom(r,&p,0,(const uint8_t*)"a\0b",3,&atom));OK(a->btt_view(r,&atom,&view));
    CHECK(view.kind==BTT_ATOM && view.length==3 && !memcmp(view.bytes,"a\0b",3));
    CHECK(a->btt_atom(r,&p,0,(uint8_t[]){255},1,&value)==BTT_UNSUPPORTED);
    OK(a->btt_value(r,&p,BTT_NIL,NULL,0,NULL,0,&nil));
    BttValue fields[]={atom,nil};OK(a->btt_value(r,&p,BTT_CONS,NULL,0,fields,2,&cons));
    OK(a->btt_value(r,&p,BTT_TUPLE,NULL,0,fields,2,&tuple));
    OK(a->btt_value(r,&p,BTT_MAP,NULL,0,fields,2,&map));
    OK(a->btt_key(r,&map,0,&value));OK(a->btt_view(r,&value,&view));CHECK(view.kind==BTT_ATOM);OK(a->btt_release(r,&value));
    BttValue roots[]={atom,nil,cons,tuple,map};for(unsigned i=0;i<5;i++)OK(a->btt_release(r,&roots[i]));

    for(unsigned kind=1;kind<=4;kind++) {
        BttCallable info;OK(a->btt_opaque(r,&p,0,100+kind,kind,&value));
        OK(a->btt_view(r,&value,&view));CHECK(view.kind==BTT_OPAQUE);
        OK(a->btt_callable_info(r,&value,&info));CHECK(info.kind==BTT_HOST_OPAQUE && info.host_id==100+kind);
        BttCopy job;BttValue out={0};uint32_t work;int status;unsigned turns=0;
        OK(a->btt_copy_begin(r,&to,&value,&job));
        while((status=a->btt_copy_step(r,&job,1,&out,&work))==BTT_OK)CHECK(work<=1 && ++turns<10000);
        CHECK(status==BTT_UNSUPPORTED && !out.owner);OK(a->btt_copy_cancel(r,&job));turns=0;
        while((status=a->btt_copy_step(r,&job,1,&out,&work))==BTT_OK)CHECK(work<=1 && ++turns<10000);
        CHECK(status==BTT_CANCELLED && !out.owner);OK(a->btt_release(r,&value));
    }
    TRY(finish(a,r,&p));TRY(finish(a,r,&to));return 0;
}

EXPORT int beaver_batata_transactions(void **functions,const char *identity) {
    BttTestAPI a;memcpy(&a,functions,sizeof(a));TRY(lifecycle(&a,identity));BttQuery query;OK(a.btt_query(BTT_ABI_ID,&query));CHECK(!strcmp(query.identity,identity));
    BttRuntime *r=NULL,*foreign=NULL;OK(a.btt_create(BTT_ABI_ID,&r));OK(a.btt_create(BTT_ABI_ID,&foreign));
    BttEntry entry={0,0};TRY(imports_and_copy(&a,r,foreign,&entry));TRY(rollback_and_references(&a,r,&entry));
    TRY(callables_and_events(&a,r));TRY(constructors_and_capabilities(&a,r,&entry));
    OK(a.btt_destroy(r));OK(a.btt_destroy(foreign));return 0;
}
