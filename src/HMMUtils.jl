
function h2vtable(m, postag::String)
        ih = findfirst(isequal(postag), m.tags)
        if isnothing(ih) error("$(postag) not found in ctb") end
        vs = collect(m.h2v[ih])
        sort!(vs; by=last, rev=true)
        nv = length(vs)
        mat = Matrix{Any}(undef, (nv, 3))
        word_ids, probs = map(first, vs), exp.(map(last, vs))
        mat[:,1] = m.words[word_ids]
        mat[:,2] = map(p -> trunc(p, digits=6), probs)
        mat[:,3] = map(i -> (i > length(m.words) - m.usr_words ? "user.dict" : "CTB"), word_ids)
        return UselessTable(mat; cnames=["word", "prob.", "source"], foots=["$(postag) has $(length(vs)) words", "Unknown words porb. = $(1. - sum(probs))"])
end

function v2htable(m, word::String)
        word_id = findfirst(isequal(word), m.words)
        if isnothing(word_id) error("$(word) not found in m") end
        np = length(m.tags)
        hs = [(m.tags[first(p)], haskey(last(p), word_id) ? exp(last(p)[word_id]) : 0.) for p in enumerate(m.h2v)]
        sort!(hs; by=last, rev=true)
        mat = Matrix{Any}(undef, (np, 1))
        mat[:,1] = map(d -> trunc(d, digits=6), map(last, hs))
        return UselessTable(mat; cnames=["Prob."], rnames=map(first, hs), topleft="From\\To")
end

function h2htable(m)
        np = length(m.tags)
        mat = map(d->Int(round(exp(d)*100)), m.h2h)
        return UselessTable(mat; cnames=m.tags, rnames=m.tags, topleft="From\\To(%)")
end

function hprtable(m)
        np = length(m.tags)
        mat = Matrix{Any}(undef, (np, 1))
        mat[:,1] = map(d->trunc(exp(d), digits=6), m.hpr)
        return UselessTable(mat; cnames=["prob."], rnames=m.tags)
end

function postable(m)
        tsv = readdlm(joinpath(pathof(KongYiji), "..", "..", "data", "postable.tsv"), '\t', String)
        return UselessTable(tsv[2:end,:]; cnames=tsv[1,:], heads=["CTB postable"])
end

function normalize!(h2v::Vector{Dict{Int,Tv}}, INF::Vector{Tv})
        for (ih, vs) in enumerate(h2v) 
                tot = sum(values(vs)) + EPS * (length(vs) + 1)
                for (k, v) in vs
                        vs[k] = (v + EPS) / tot
                end
                INF[ih] = EPS / tot
        end
end

function normalize!(hpr::Vector{Tv})
        hpr .+= EPS
        hpr ./= sum(hpr)
end

function normalize!(h2h::Matrix{Tv})
        h2h .+= EPS
        h2h ./= sum(h2h; dims=2)
end

log!(hpr::Vector{Tv}) = hpr .= log.(hpr)
log!(h2h::Matrix{Tv}) = h2h .= log.(h2h)
function log!(h2v::Vector{Dict{Int,Tv}})
        for (ih, vs) in enumerate(h2v) 
                for (k, v) in vs
                        vs[k] = log(v)
                end
        end
        h2v
end

