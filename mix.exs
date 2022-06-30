defmodule Met.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      # Docs
      name: "MLIR",
      source_url: "https://github.com/USER/PROJECT",
      homepage_url: "http://YOUR_PROJECT_HOMEPAGE",
      docs: docs()
    ]
  end

  defp docs() do
    [
      # The main page in the docs
      # main: "MLIR",
      extras: ["README.md"],
      groups_for_modules: [
        DSL: [
          Beaver,
          Beaver.MLIR
        ],
        IR: [
          Beaver.MLIR.Attribute,
          Beaver.MLIR.Block
        ],
        JIT: [
          Beaver.MLIR.ExecutionEngine,
          Beaver.MLIR.ExecutionEngine.MemRefDescriptor
        ],
        Pass: [
          Beaver.MLIR.Pass,
          Beaver.MLIR.Pass.Composer
        ],
        Bindings: [
          Beaver.MLIR.CAPI
        ],
        Nx: [
          Beaver.Nx.Backend,
          Beaver.Nx.Compiler,
          Beaver.Nx.Defn,
          Beaver.Nx.MemrefAllocator
        ],
        Exotic: [
          Exotic,
          Exotic.NIF,
          Exotic.Value,
          Exotic.Valuable,
          Exotic.Value.Struct,
          Exotic.Value.Array
        ]
      ],
      nest_modules_by_prefix: [
        Beaver.MLIR.Dialect
      ]
    ]
  end

  # Dependencies listed here are available only for this
  # project and cannot be accessed from applications inside
  # the apps folder.
  #
  # Run "mix help deps" for examples and options.
  defp deps do
    [{:ex_doc, "~> 0.27", only: :dev, runtime: false}]
  end
end
