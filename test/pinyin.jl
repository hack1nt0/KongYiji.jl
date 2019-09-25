
using Unicode

pydict = Dict{Char, String}()
pyset = Set{String}()

for line in eachline(joinpath("D:\\", "pinyin.txt"))
        if startswith(line, "#"); continue; end
        code, pys, chr = split(line, [' ', '#', ':']; keepempty=false)
        pys = split(Unicode.normalize(pys; stripmark=true), ','; keepempty=false) |> unique
        
        for py in pys; push!(pyset, py); end

        pys = join(pys, ',')
        chr = chr[1]
        #@show pys
        pydict[chr] = pys
        #@show chr, pydict[chr]
end

@show length(pyset)

@show pyset

open(joinpath("D:\\", "chr2py.txt"), "w") do io
        for (k, v) in pydict; println(io, k, ' ', v); end
end
        

pinyin(c::Char) = get(pydict, c, c)

pinyin(s::AbstractString) = map(pinyi, s)

#pinyin(ss::Vector{T <: AbstractString}) where T = map(pinyi, ss)

        