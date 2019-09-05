using Knet
using Knet: softmax, RNN, adam, progress, mean, minibatch, KnetArray, value
using Base.Iterators: flatten
import Base.iterate
#=
struct Chain
    layers
    Chain(layers...) = new(layers)
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
=#

function (m::RNN)(x, y)
        lp = softmax(m(KnetArray(x)); dims=1)
        y = KnetArray(y)

        #@show size(x), size(y)
        #=
        r = 0.
        for b = 1:B, t = 1:length(y[b])
                r += log(sum(lp[:,b,t] .* y[b][t]))
        end
        r / B
        =#
        #-mean(log.(sum(lp .* KnetArray(y); dims=1)))
        nh, nb, nt = size(lp)
        nc = nb * nt
        lp = reshape(lp, nh, nc)
        y = reshape(y, nh, nc)
        @assert size(lp) == size(y)
        r = Float32(0)
        #=
        lp2 = Array(lp)
        if any(isequal(0), lp2)
                @show lp2
        end
        y2 = Array(y)
        if any(isequal(0), y2)
                @show y2
        end
        =#
        for i in 1:nc; r += kl(lp[:,i], y[:,i]) end
        return r / nc
end

kl(p, q) = -sum(p .* (log.(q) - log.(p)))
        
mutable struct InfHMM
        dict
        pmp
        vectors    #h2v
        nn
end

function InfHMM(ctb; batchsz=256, seqlen=100, epochs=1, file=joinpath(pathof(KongYiji), "..", "..", "data", "InfHMM-model.jld2"))
        wmp, pmp, ppr = Dict{String,Int}(), Dict{String,Int}(), DefaultDict{String, Int}(0)
        for doc in ctb, (is, sent) in enumerate(doc), (pos, word) in sent
                wmp[word] = get(wmp, word, length(wmp) + 1)
                pmp[pos ] = get(pmp, pos , length(pmp) + 1)
                if is == 1 ppr[pos] += 1 end
        end
        np = length(pmp)
        hpr = fill(Float32(0), np)
        for (pos, cnt) in ppr hpr[pmp[pos]] = cnt end
        hpr ./= sum(hpr)

        dict = AhoCorasickAutomaton{UInt32}(wmp)

        nw = length(wmp)
        vectors = fill(Float32(1e-9), np, nw)
        for doc in ctb, sent in doc, (pos, word) in sent vectors[pmp[pos],wmp[word]] += 1 end
        
        sents = collect(flatten(ctb))
        batchsz = min(batchsz, length(sents))
        nn = RNN(np, np; h=KnetArray(repeat(hpr, 1, batchsz, 1)), rnnType=:lstm, dataType=Float32, skipInput=true, bidirectional=false)
        #nn = RNN(1, np; h=nothing, rnnType=:lstm, dataType=Float32, skipInput=true, bidirectional=false)

        #@show size(nn.h)
        @show summary(nn)
        #@show typeof(vec(value(nn.h)))
        
        ret = InfHMM(dict, pmp, vectors, nn)
        train!(ret, sents, batchsz, seqlen, epochs, file; update=false)

        return ret
end

function train!(m::InfHMM, sents, batchsz, seqlen, epochs, file; update=true)
        if update
                wmp, pmp = Dict(collect(m.dict)), m.pmp
                oldnw = length(wmp)
                for sent in sents, (pos, word) in sent
                        wmp[word] = get(wmp, word, length(wmp) + 1)
                        haskey(m.pmp, pos) || error("$pos not existed in ctb tagset")
                end
                newnw, np = length(wmp), length(pmp)
                newvectors = fill(Float32(1e-9), np, nw)
                for sent in sents, (pos, word) in sent newvectors[pmp[pos],wmp[word]] += 1 end
                cmp = newnw - oldnw
                if cmp == 0 
                        m.vectors .+= vectors
                elseif cmp > 0
                        m.dict = AhoCorasickAutomaton{UInt32}(wmp)
                        newvectors[:,1:oldnw] .+= m.vectors
                        m.vectors = newvectors
                else
                        m.vectors[:,1:newnw] .+= newvectors
                end
        end
        
        wsents = [[m.dict[word] for (_, word) in sent] for sent in sents]
        ys = d3(wsents, normalize(m.vectors), batchsz, 1. / size(m.vectors, 1))
        xs = d3(wsents, zeros(Float32, size(m.vectors)), batchsz, 0)
        seqlen = min(seqlen, size(ys, 3))
        @show summary(m.vectors)
        @show summary(xs)
        @show summary(ys)
        seqlen = min(seqlen, size(ys, 3))
        data = minibatch(xs, ys, seqlen; shuffle=true)
        @show summary.(first(data))
        #shuffle!(data)
        for x in progress(adam(m.nn, data))
                print("\n loss: ", x, "\n")
        end
        if isfile(file); Knet.save(file, "model", m) end
        Knet.gc() # To save gpu memory
end

normalize(m::Matrix) = m ./ sum(m; dims=1)

function d3(sents, vectors, batchsz, pad)
        ns = length(sents)
        bs = div(ns + batchsz - 1, batchsz)
        tword = eltype(sents[1])
        batches = [tword[] for _ in 1:batchsz]
        for b in 1:batchsz, s in (b - 1) * bs + 1:min(ns, b * bs)
                append!(batches[b], sents[s])
        end
        @show summary(batches)
        seqlen_max = mapreduce(length, max, batches)
        inputsz = size(vectors, 1)
        ret = zeros(Float32, inputsz, batchsz, seqlen_max)
        for t in 1:seqlen_max, b in 1:batchsz
                ret[:,b,t] .= t <= length(batches[b]) ? vectors[:,batches[b][t]] : pad
        end
        return ret
end

struct Score_Word
        precision
        recall
        oov_recall
end

function (hmm::InfHMM)(x::String, dp::Matrix{Float32}, pre::Matrix{Tuple{Int, Int}}, pprob::Matrix{Float32}; recover=true)
        origin = x
        x = normalize_width_numeric_space(x)
        chrs = codeunits(x)
        vtxs = collect(eachmatch(hmm.dict, chrs))
        sort!(vtxs)
        nv, nc = length(vtxs), length(chrs)
        fill!(dp, Inf)
        #===
        @show hmm
        @show maximum(dp[nc + 1, :])
        @show dp
        @show vtxs
        =###
        v = (nc + 1, argmax(dp[nc + 1,:]))
        ret = fill("", 0)
        while v[1] != 1
                pv = pre[v[1],v[2]]
                push!(ret, x[pv[1]:prevind(x, v[1])])
                v = pv
        end
        reverse!(ret)
        if recover ret = denormalize_width_numeric_space(ret, origin) end
        return ret
end

function (hmm::InfHMM)(xs::Vector{String}; hist=100)
        nc_max = mapreduce(ncodeunits, max, xs)
        dp = fill(-Inf, (nc_max + 1, hist))
        pre = fill((1, 0), (nc_max + 1, hist))
        np = length(hmm.pmp)
        pprob = zeros(0., np, hist)
        return [hmm(x, dp, pre, pprob) for x in xs]
end

function (hmm::InfHMM)(x::String; hist=100)
        return hmm([x]; hist=hist)[1]
end
