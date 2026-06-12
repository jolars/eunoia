# Input parsing and inclusive/exclusive conversion, ported from the `eunoia-py`
# sister package (`_parse.py`). These are pure helpers: the native library still
# does the spec building and fitting; this code shapes the user's input into
# `(combination, size)` pairs and reconstructs fitted areas in the user's scale.

# Value types that mark a membership-list input (vs. a region area).
const _MEMBERSHIP_TYPES = Union{AbstractVector,AbstractSet,Tuple}

"""
    canonicalize(combo) -> String

Return `combo` in canonical form: trim parts, drop empties, sort, rejoin with
`&`. Matches the eunoia core's `Combination` display, so canonical keys line up
with the combination strings the native library returns.
"""
function canonicalize(combo::AbstractString)
    parts = sort!(filter(!isempty, [String(strip(s)) for s in split(combo, '&')]))
    isempty(parts) && error("invalid_combination: $(repr(combo))")
    return join(parts, "&")
end

"""
    is_membership_input(input) -> Bool

`true` when `input` maps set names to membership collections (vector/set/tuple)
rather than to region areas. Errors on a *mixed* mapping (some collections, some
not). An empty mapping is treated as area input (`false`).
"""
function is_membership_input(input::AbstractDict)
    isempty(input) && return false
    flags = [isa(v, _MEMBERSHIP_TYPES) for v in Base.values(input)]
    all(flags) && return true
    any(flags) && error(
        "invalid_input: mix of membership collections and non-collections; " *
        "values must be all areas or all membership lists")
    return false
end

"""
    parse_membership_input(input) -> Dict{String,Float64}

Convert membership lists to exclusive per-region counts. Each element is
assigned to the canonical combination of the sets it belongs to, deduplicated
within a set and stringified, then counted per region.
"""
function parse_membership_input(input::AbstractDict)
    membership = Dict{String,Set{String}}()
    for (set_name, members) in input
        for element in Set(members)
            push!(get!(membership, string(element), Set{String}()), String(set_name))
        end
    end
    counts = Dict{String,Float64}()
    for sets in Base.values(membership)
        isempty(sets) && continue
        combo = canonicalize(join(sort!(collect(sets)), "&"))
        counts[combo] = get(counts, combo, 0.0) + 1.0
    end
    return counts
end

"""
    to_inclusive(fitted_exclusive, keys) -> Dict{String,Float64}

For each key `X`, sum the fitted exclusive areas of every region that is a
superset of `X`. Used to express fitted areas in the user's input scale when
`input_type="inclusive"`.
"""
function to_inclusive(fitted_exclusive, keys)
    _sets(s) = Set(filter(!isempty, [String(strip(p)) for p in split(s, '&')]))
    result = Dict{String,Float64}()
    for k in keys
        x_sets = _sets(k)
        total = 0.0
        for (combo, val) in fitted_exclusive
            issubset(x_sets, _sets(combo)) && (total += val)
        end
        result[k] = total
    end
    return result
end

# ---------------------------------------------------------------------------
# venn name resolution
# ---------------------------------------------------------------------------

"""Default set name for the `i`-th set (1-based): `A`, `B`, …, then `set27`…"""
_default_name(i::Integer) = i <= 26 ? string('A' + (i - 1)) : "set$i"

"""
    _resolve_names(sets) -> Vector{String}

Resolve `venn`'s input to a list of set names: an `Integer` `n` → default names;
a vector of names; or a mapping whose keys' base set names are extracted.
"""
function _resolve_names(sets)
    # Bool <: Integer in Julia; reject it explicitly to avoid `venn(true)`.
    if isa(sets, Bool)
        throw(ArgumentError(
            "venn: 'sets' must be an Integer, a vector of names, or a mapping"))
    elseif isa(sets, Integer)
        sets < 1 && throw(ArgumentError("venn: number of sets must be >= 1"))
        return [_default_name(i) for i in 1:sets]
    elseif isa(sets, AbstractDict)
        names = String[]
        for key in keys(sets)
            for part in split(canonicalize(string(key)), '&')
                p = String(part)
                (!isempty(p) && !(p in names)) && push!(names, p)
            end
        end
        isempty(names) && throw(ArgumentError("venn: no sets found in mapping"))
        return names
    elseif isa(sets, AbstractString)
        throw(ArgumentError("venn: pass a vector of set names, not a single string"))
    else
        names = [string(s) for s in sets]
        isempty(names) && throw(ArgumentError("venn: need at least one set name"))
        length(Set(names)) != length(names) &&
            throw(ArgumentError("venn: set names must be unique"))
        return names
    end
end
