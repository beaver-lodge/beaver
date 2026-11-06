// ABI declarations from wavefrontier/batata aff71ee; independent test fixture.
#ifndef BATATA_ABI_H
#define BATATA_ABI_H
#include <stdint.h>
#define BTT_ABI_ID UINT64_C(0x425454000000000c)
typedef struct BttRuntime BttRuntime;
typedef struct {uint32_t program,index;} BttEntry;
typedef struct {uint64_t owner,generation;uint32_t process,slot;} BttValue;
typedef struct {uint64_t owner,generation;uint32_t slot,reserved;} BttProcess;
typedef struct {uint64_t owner,generation;uint32_t process,reserved;} BttCopy;
typedef struct {uint64_t owner,process_generation,generation;uint32_t process,reserved;} BttImport;
enum {BTT_OK,BTT_INVALID,BTT_UNSUPPORTED,BTT_BUSY,BTT_FAILED,BTT_WAITING,BTT_DONE,BTT_LIMIT,BTT_CANCELLED};
enum {BTT_ATOM,BTT_BINARY,BTT_NIL,BTT_CONS,BTT_TUPLE,BTT_MAP,BTT_CALLBACK,BTT_INTEGER,BTT_FLOAT,BTT_OPAQUE};
enum {BTT_HOST_CALLABLE,BTT_GUEST_CLOSURE,BTT_GUEST_EXPORT,BTT_HOST_OPAQUE};
typedef struct {uint32_t kind,program,node,arity,captures,host_id;const char *program_identity,*identity;} BttCallable;
typedef struct {uint64_t abi;uint32_t programs,capabilities;const char *identity;} BttQuery;
typedef struct {uint32_t state,work;uint64_t reductions;} BttStep;
typedef struct {uint32_t kind,count;const uint8_t *bytes;uint64_t length;} BttView;
enum {BTT_EVENT_CALLBACK,BTT_EVENT_PRIMITIVE};
enum {BTT_STRING_PRINTABLE=1,BTT_FLOAT_STRING,BTT_INSPECT_FLOAT,BTT_STRING_FLOAT,BTT_INSPECT_BINARY,BTT_BINARY_TO_ATOM,BTT_BINARY_TO_EXISTING_ATOM,BTT_EXCEPTION_NORMALIZE,BTT_INSPECT_OPTIONS,BTT_ERROR_METADATA,BTT_FLOAT_TO_BINARY,BTT_CHARLIST_STRING,BTT_INSPECT_PLAIN,BTT_ENSURE_COMPILED,BTT_PROTOCOL_CONCAT,BTT_PROTOCOL_EXCEPTION,BTT_DATE_ISO8601,BTT_TIME_ISO8601,BTT_NAIVE_DATETIME_ISO8601,BTT_DATETIME_ISO8601};
#define BTT_CAP_STRING_PRINTABLE UINT64_C(1)
#define BTT_CAP_FORMATTING UINT64_C(2)
#define BTT_CAP_ATOMS UINT64_C(4)
#define BTT_CAP_ERRORS UINT64_C(8)
#define BTT_CAP_CALLER_SERVICES UINT64_C(16)
typedef struct {uint64_t request;BttProcess process;uint32_t kind,operation,callback,arity;BttValue arguments[16];} BttEvent;
#define BTT_TEST_VALUE(type) type
#define BTT_TEST_CALL(name, ...) int btt_##name(__VA_ARGS__);
#include "api.def"
#undef BTT_TEST_CALL
#undef BTT_TEST_VALUE
#endif
