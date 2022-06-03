defmodule Exotic.Application do
  def start(_type, _args) do
    children = [Exotic.LibC.Managed]
    Supervisor.start_link(children, strategy: :one_for_one)
  end
end
