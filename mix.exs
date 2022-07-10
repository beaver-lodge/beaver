defmodule Beaver.Umbrella.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "MLIR",
      source_url: "https://github.com/beaver-project/beaver",
      homepage_url: "https://hexdocs.pm/beaver",
      docs: docs()
    ]
  end

  defp docs() do
    [
      main: "Beaver",
      extras: ["README.md", "apps/exotic/README.md"],
      filter_modules: fn m, _meta ->
        name = Atom.to_string(m)

        not (String.contains?(name, "Beaver.MLIR.Dialect.") ||
               String.contains?(name, "Beaver.MLIR.CAPI."))
      end,
      api_reference: false,
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
      ]
    ]
  end

  defp deps do
    [{:ex_doc, "~> 0.27", only: :dev, runtime: false}]
  end
end
