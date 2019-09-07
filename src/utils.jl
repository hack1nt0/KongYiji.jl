
function halfwidth(c::Char)
        code = UInt(c)
        if 0xFF01 <= code <= 0xFF5E
                return Char(code - 0xFF01 + 0x0021)
        else
                return c
        end
end

function fullwidth(c::Char)
        code = UInt(c)
        if 0x0021 <= code <= 0x007E
                return Char(code - 0x0021 + 0xFF01)
        else
                return c
        end
end

const CH_NUMBERS = "零一二三四五六七八九十百千万亿"

function normalize_width_numeric(c::Char)
        c = halfwidth(c)
        return isnumeric(c) || any(isequal(c), CH_NUMBERS) ? 'N' : c
end

# normalize width chars & numeric chars & spaces
mask(x::AbstractString) = filter(!isspace, map(normalize_width_numeric, x))

# inverse function of mask
function demask(y::Vector{String}, x::String)
        xchrs = collect(x)
        nx = length(xchrs)
        projections = Vector{Char}(undef, nx)
        ip = 1
        for ix in 1:nx
                if isspace(xchrs[ix]) continue end
                projections[ip] = xchrs[ix]
                ip += 1
        end
        ret = Vector{String}(undef, length(y))
        ip = 1
        for (iy, word) in enumerate(y)
                nw = length(word)
                new_w = Vector{Char}(undef, nw)
                for iw in 1:nw
                        new_w[iw] = projections[ip]
                        ip += 1
                end
                ret[iy] = String(new_w)
        end
        return ret
end

#=
str = "  1999年10月5日 一八九九年 "
println(map(halfwidth, str))
println(normalize_width_numeric_space(str))
=#

isevil(c::Char) = isspace(c) || isnumeric(c) || isletter(c) || ispunct(c)
isevil(s::String) = all(isevil, s)
filter_evil(s::String) = String(filter(!isevil, s))

function shrink_evil(s::String)
        chrs = collect(s)
        nchr = length(chrs)
        ret = Char[]
        sizehint!(ret, nchr)
        i = 1
        while i <= nchr
                while i <= nchr && !isevil(chrs[i])
                        push!(ret, chrs[i])
                        i += 1
                end
                s = i
                while i <= nchr && isevil(chrs[i]) i += 1 end
                t = i
                while s < t && isspace(chrs[s]) s += 1 end
                while s < t && isspace(chrs[t - 1]) t -= 1 end
                if s < t append!(ret, 'E') end
        end
        return String(ret)
end

function expand_evil(y::Vector{String}, x::String)::Vector{String}
        xchrs = collect(x)
        nx = length(xchrs)
        projections = Vector{UnitRange{Int64}}(undef, length(xchrs))
        ix, iy = 1, 1
        while ix <= nx
                while ix <= nx && !isevil(xchrs[ix])
                        projections[iy] = ix:ix
                        ix += 1
                        iy += 1 
                end
                s = ix
                while ix <= nx && isevil(xchrs[ix]) ix += 1 end
                t = ix
                while s < t && isspace(xchrs[s]) s += 1 end
                while s < t && isspace(xchrs[t - 1]) t -= 1 end
                if s < t
                        projections[iy] = s:t-1
                        iy += 1
                end
        end                
        ret = Vector{String}(undef, length(y))
        offset = 0
        for (i, word) in enumerate(y)
                new_w = Char[]
                sizehint!(new_w, length(word))
                for _ in 1:length(word)
                        offset += 1
                        append!(new_w, x[projections[offset]])
                end
                ret[i] = String(new_w)
        end                
        return ret
end

# traditional to simple chinese character
function simplify(c::Char)
    return c
end

simplify(s::String) = String(simplify.(collect(s)))


dir(o...) = joinpath(pathof(KongYiji), "..", "..", "data", o...)


function getid!(d::Dict{String,Ti}, k::String)
        if haskey(d, k) 
                return d[k]
        else
                v = length(d) + 1
                d[k] = v
                return v
        end
end

function vec(d::Dict{String, Ti})
        r = Array{String}(undef, length(d))
        for (k, v) in d r[v] = k end
        return r
end

# Assume the unique of elements of v
function dict(v::Vector{String})
        r = Dict{String, Ti}()
        for (i, k) in enumerate(v) r[k] = i end
        return r
end
