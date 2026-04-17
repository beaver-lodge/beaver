[
  verifier_graph: [
    [
      kind: :compile,
      command: "sh",
      args: [
        "-lc",
        "PATH=/opt/homebrew/opt/zig@0.15/bin:/Users/tsai/Downloads/llvm-install/bin:$PATH " <>
          "SDKROOT=$(xcrun --show-sdk-path) " <>
          "LLVM_CONFIG_PATH=/Users/tsai/Downloads/llvm-install/bin/llvm-config " <>
          "mix compile --warnings-as-errors"
      ]
    ],
    [
      kind: :test,
      command: "sh",
      args: [
        "-lc",
        "PATH=/opt/homebrew/opt/zig@0.15/bin:/Users/tsai/Downloads/llvm-install/bin:$PATH " <>
          "SDKROOT=$(xcrun --show-sdk-path) " <>
          "LLVM_CONFIG_PATH=/Users/tsai/Downloads/llvm-install/bin/llvm-config " <>
          "mix test"
      ]
    ],
    [
      kind: :dialyzer,
      timeout_ms: 900_000,
      command: "sh",
      args: [
        "-lc",
        "PATH=/opt/homebrew/opt/zig@0.15/bin:/Users/tsai/Downloads/llvm-install/bin:$PATH " <>
          "SDKROOT=$(xcrun --show-sdk-path) " <>
          "LLVM_CONFIG_PATH=/Users/tsai/Downloads/llvm-install/bin/llvm-config " <>
          "mix dialyzer --quiet --format dialyxir"
      ]
    ]
  ],
  reach: [
    enabled: true,
    mode: :project,
    frontend: :source,
    paths: ["lib/**/*.ex"],
    cache_path: ".restraint/reach",
    contract_center_limit: 5,
    locality_limit: 6,
    satellite_limit: 7
  ],
  type_gate_baseline_path: ".dialyzer_ignore.exs",
  dirty_repo_mode: :refuse,
  constitution_path: "CONSTITUTION.md",
  workspace_topology: :preserve_parent
]
