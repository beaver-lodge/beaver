defmodule Beaver.SSA.Scalar do
  @moduledoc "Signed 64-bit interval and boolean facts with widening, bottom, and unknown values."
  import Bitwise
  @lo -9_223_372_036_854_775_808
  def join(left, right, widen?), do: join_f(left, right, widen?)
  def infer(operation, arguments), do: inf_ar(operation, arguments)
  @doc "Infers a scalar instruction using caller-named literal, truth, shape, byte and arithmetic operations."
  def instruction(op, arguments, results, vocabulary) do
    {name, value} = if is_map(op), do: hd(Map.to_list(op)), else: {op, nil}
    kind = Map.get(vocabulary, name, :unknown)

    cond do
      :bottom in arguments -> List.duplicate(:bottom, results)
      kind == :byte -> [{:integer, 0, 255}, {:boolean, nil}]
      results == 1 -> [fact(kind, value, arguments, List.first(arguments, :any))]
      true -> List.duplicate(:any, results)
    end
  end

  defp fact(:atom, s, _, _) when s in ["true", "false"], do: {:boolean, s == "true"}
  defp fact(:truth, _, _, f), do: {:boolean, if(f != :any, do: if(f, do: tuple_size(f) > 2 || elem(f, 1), else: false))}
  defp fact(:shape, %{"kind" => k}, _, f), do: {:boolean, shape(k, f)}
  defp fact(:arithmetic, name, arguments, _), do: infer(name, arguments)
  defp fact(:integer, n, _, _) when n not in [nil, false], do: {:integer, n, n}
  defp fact(nil, _, _, _), do: nil
  defp fact(_, _, _, _), do: :any
  defp shape("boolean", {:boolean, _}), do: true
  defp shape(k, {:boolean, b}) when k in ~w(true false) and is_boolean(b), do: b == (k == "true")
  defp shape(k, x), do: if(x != :any, do: k == "nil" and is_nil(x))

  defp jf(a, b, c, d, w), do: {:integer, ((w and c < a) && @lo) || min(a, c), ((w and d > b) && bnot(@lo)) || max(b, d)}
  defp join_f(a, b, _) when a == :bottom or b == :bottom or a == b, do: if(a == :bottom, do: b, else: a)
  defp join_f({:integer, a, b}, {:integer, c, d}, w), do: jf(a, b, c, d, w)
  defp join_f(a, b, _), do: if(match?({{:boolean, _}, {:boolean, _}}, {a, b}), do: {:boolean, nil}, else: :any)
  defp ivl(a, b), do: if(a in @lo..bnot(@lo) and b in @lo..bnot(@lo), do: {:integer, a, b}, else: :any)
  defp ivl_l(xs), do: ivl(Enum.min(xs), Enum.max(xs))
  defp inf_ar("sub", [{:integer, a, b}]), do: ivl(-b, -a)
  defp inf_ar(o, [{:integer, a, b}, {:integer, c, d}]), do: iar(o, a, b, c, d)
  defp inf_ar(_, _), do: :any
  defp iar(o, a, a, c, c) when o in ~w(div rem) and c != 0 and {a, c} != {@lo, -1}, do: idr(o, a, c)
  defp iar(o, a, a, c, c) when o in ~w(band bor), do: ivl_l([(o == "bor" && (a ||| c)) || (a &&& c)])
  defp iar(o, a, b, c, d) when o in ~w(add sub), do: if(o == "add", do: ivl(a + c, b + d), else: ivl(a - d, b - c))
  defp iar("mul", a, b, c, d), do: ivl_l([a * c, a * d, b * c, b * d])
  defp iar("bsl", a, b, c, c) when c in 0..63, do: ivl(a <<< c, b <<< c)
  defp iar("bsr", a, b, c, c) when c in 0..63, do: {:integer, a >>> c, b >>> c}
  defp iar("band", a, b, c, d) when a >= 0 and c >= 0, do: {:integer, 0, min(b, d)}
  defp iar(o, _, _, _, _) when o in ~w(band bor), do: {:integer, @lo, bnot(@lo)}
  defp iar(o, a, b, c, d) when o in ~w(eq strict_eq ne strict_ne), do: ce(a, b, c, d, o in ~w(eq strict_eq))
  defp iar(o, a, b, c, d) when o in ~w(lt le gt ge), do: {:boolean, cmp_ord(o, a, b, c, d)}
  defp iar(_, _, _, _, _), do: :any
  defp idr(m, a, c), do: ivl_l([if(m == "div", do: div(a, c), else: rem(a, c))])
  defp ce(a, b, c, d, q), do: {:boolean, if(b < c or d < a, do: !q, else: if(a == b and c == d, do: a == c == q))}
  defp cmp_ord(o, a, b, c, d), do: co2(o in ~w(le ge), if(o in ~w(gt ge), do: {c, d, a, b}, else: {a, b, c, d}))
  defp co2(e, {a, b, c, d}), do: if((e and b <= c) or b < c, do: true, else: if(a > d or (!e and a == d), do: false))
end
