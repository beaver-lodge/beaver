#ifndef BEAVER_EXECUTION_H
#define BEAVER_EXECUTION_H
#include <memory>

namespace beaver {
struct ExecutionState;
using Execution = std::shared_ptr<ExecutionState>;
Execution currentExecution();
class ExecutionScope {
public:
  explicit ExecutionScope(Execution execution);
  ~ExecutionScope();

private:
  Execution previous;
};
} // namespace beaver
#endif
