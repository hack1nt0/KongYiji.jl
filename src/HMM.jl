


mutable struct HMM
    dict::AhoCorasickAutomaton{Ti}
    words::Vector{String}
    usr_words::Int
    tags::Vector{String}
    pmp::Dict{String,Int}
    hpr::Vector{Tv}
    h2h::Matrix{Tv}
    h2v::Vector{Dict{Int, Tv}}
    INF::Vector{Tv}
    usrh2v::Vector{Dict{Int, Tv}}
    usrINF::Vector{Tv}
    usrw::Tv
end

function HMM(;usrw=0.5, EPS=Tv(1e-9), base=nothing)
        @assert usrw <= 1.
        @assert base isa HMM || base isa Nothing

        if base == nothing
                file = KongYiji.dir("hmm.jld2")
                if !isfile(file) file = decompress(KongYiji.dir("hmm.jld2.7z")) end
                @assert isfile(file)
                r = Knet.load(file, "hmm")
        else
                r = base
        end

        r.usrw = usrw
        usrdict = KongYiji.dir("usrdict.txt")
        wmp = KongYiji.dict(r.words)
        for line in eachline(usrdict)
                 cells = split(line)
                 pos, word = "", ""
                 if length(cells) >= 2 pos, word = cells[2], cells[1] end
                 if length(cells) == 1 word = cells[1] end
                 if pos != "" && !haskey(r.pmp, pos) @warn ("Postag $(pos) not defined, ignore."); continue end
                 if word == "" continue end
                 if pos == "" pos = "NN" end  #NOTE default pos NR
                 word = mask(word)
                 if !haskey(wmp, word) wmp[word] = length(wmp) + 1; push!(r.words, word); r.usr_words += 1 end
                 iw, ip = wmp[word], r.pmp[pos]
                 r.usrh2v[ip][iw] = get(r.usrh2v[ip], iw, 0) + 1
        end                        
        normalize!(r; EPS=EPS)
        log!.((r.hpr, r.h2h))
        r.dict = AhoCorasickAutomaton{Ti}(r.words)
        r
end

function HMM(corpus)

        tags, words = KongYiji.postags(corpus), KongYiji.words(corpus)
        np, pmp = length(tags), KongYiji.dict(tags)
        
        hpr, h2h, h2v, INF = fill(Tv(0), np), fill(Tv(0), (np, np)), [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        usrh2v, usrINF, usrw = [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np), Tv(0)
        for sent in sents(corpus)
                pp = 0
                for (ip, iw) in sent
                        if pp == 0 hpr[ip] += 1 else h2h[pp,ip] += 1 end
                        pp = ip
                        h2v[ip][iw] = get(h2v[ip], iw, 0) + 1
                end
        end
        dict = AhoCorasickAutomaton{Ti}()
        HMM(dict, words, 0, tags, pmp, hpr, h2h, h2v, INF, usrh2v, usrINF, usrw)
end

function train!(hmm::HMM, corpus)
        tags, words = KongYiji.postags(corpus), KongYiji.words(corpus)
        @assert length(setdiff(tags, hmm.tags)) == 0
        pmp = KongYiji.dict(tags)
        wmp = dict(hmm.words)
        nw = length(wmp)

        for sent in sents(corpus)
                pp = 0
                for (ip, iw) in sent
                        ip = pmp[tags[ip]]
                        if pp == 0 hmm.hpr[ip] += 1 else hmm.h2h[pp,ip] += 1 end
                        pp = ip

                        iw = getid!(wmp, words[iw])
                        hmm.h2v[ip][iw] = get(hmm.h2v[ip], iw, 0) + 1
                end
        end
        if nw < length(wmp)
                hmm.words = vec(wmp)
                hmm.dict = AhoCorasickAutomaton{Ti}(wmp)
        end
        hmm
end


function normalize!(hmm::HMM; EPS=Tv(1e-9))
        xs = hmm.hpr
        xs .+= EPS;
        xs ./= sum(xs)
        xs = hmm.h2h
        xs .+= EPS;
        xs ./= sum(xs; dims=2)

        for (ih, vs) in enumerate(hmm.h2v) 
                tot = sum(values(vs)) + EPS * (length(vs) + 1)
                for (k, v) in vs
                        vs[k] = (v + EPS) / tot #todo race condition?
                end
                hmm.INF[ih] = EPS / tot
        end
        for (ih, vs) in enumerate(hmm.usrh2v) 
                tot = sum(values(vs)) + EPS * (length(vs) + 1)
                for (k, v) in vs
                        vs[k] = (v + EPS) / tot #todo race condition?
                end
                hmm.usrINF[ih] = EPS / tot
        end

        return hmm
end

function log!(hmm::HMM)
        xs = hmm.hpr
        xs .= log.(xs)
        xs = hmm.h2h
        xs .= log.(xs)

        for (ih, vs) in enumerate(hmm.h2v) 
                for (k, v) in vs
                        vs[k] = log(v)
                end
        end
        xs = hmm.INF
        xs .= log.(xs)

        return hmm
end

log!(xs::Vector{Tv}) = xs .= log.(xs)
log!(xs::Matrix{Tv}) = xs .= log.(xs)

function (hmm::HMM)(xs::Vector{String}; recover=true, withpos=false)
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(hmm.hpr)
        @assert np > 0
        dp = fill(Tv(-Inf), (nc_max + 1, np))
        pre = fill((1, 0), (nc_max + 1, np))
        return [hmm(x, dp, pre, recover, withpos) for x in xs]
end

function (hmm::HMM)(x::String; recover=true, withpos=false)
        return hmm([x]; recover=recover, withpos=withpos)[1]
end

function (hmm::HMM)(x::String, dp::Matrix{Tv}, pre::Matrix{Tuple{Int, Int}}, recover, withpos)::Vector{String}
        origin = x
        x = mask(x)
        chrs = codeunits(x) #todo slow?
        vtxs = collect(eachmatch(hmm.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length.((vtxs, chrs, hmm.hpr))
        for i = 1:nc + 1, j in 1:np dp[i,j] = -Inf end
        pv = 1
        dp[1, :] = hmm.hpr
        pre_i = 1
        for i in 1:nc + 1
                if dp[i,1] != -Inf pre_i = i end
                while pv <= nv && vtxs[pv].s < i pv = pv + 1 end
                if !(pv <= nv && vtxs[pv].s == i || i == nc + 1) continue end
                if dp[i, 1] == -Inf
                        #@show i, pre_i
                        for pi = 1:np, pj = 1:np
                                maybe = dp[pre_i, pj] + hmm.h2h[pj, pi] + logph2v(hmm, pj, 0)
                                if maybe > dp[i, pi] dp[i, pi] = maybe; pre[i, pi] = (pre_i, pj) end
                        end
                end
                while pv <= nv && vtxs[pv].s == i
                        vtx = vtxs[pv]
                        j = i + length(vtx)
                        for pi = 1:np
                                for pj = 1:np
                                        maybe = dp[i, pi] + hmm.h2h[pi, pj] + logph2v(hmm, pi, vtx.i)
                                        if maybe > dp[j,pj] dp[j,pj] = maybe; pre[j,pj] = (i, pi) end
                                end
                        end
                        pv = pv + 1
                end
        end
        #===
        @show hmm
        @show maximum(dp[nc + 1, :])
        @show dp
        @show vtxs
        =###
        v = (nc + 1, argmax(dp[nc + 1,:]))
        words = fill("", 0)
        while v[1] != 1
                pv = pre[v[1],v[2]]
                word = x[pv[1]:prevind(x, v[1])]
                push!(words, word)
                v = pv
        end
        reverse!(words)
        if recover; words = demask(words, origin); end
        ret = words

        if withpos
                v = (nc + 1, argmax(dp[nc + 1,:]))
                postags = fill("", 0)
                while v[1] != 1
                        pv = pre[v[1],v[2]]
                        postag = hmm.tags[pv[2]]
                        push!(postags, postag)
                        v = pv
                end
                reverse!(postags)
                ret = [join(p, "/") for p in zip(words, postags)]
        end

        return ret
end

logph2v(hmm::HMM, p, v) = ((1. - hmm.usrw) * get(hmm.h2v[p], v, hmm.INF[p]) + hmm.usrw * get(hmm.usrh2v[p], v, hmm.usrINF[p])) |> log

function ==(a::HMM, b::HMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(HMM))
end

