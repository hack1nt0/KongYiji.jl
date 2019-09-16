
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

function HMM(corpus, dictpath, usrw)
        ispath(dictpath) || @warn "dictpath is not a valid path"
        @assert usrw < 1.

        tags, words = KongYiji.postags(corpus), KongYiji.words(corpus)
        np, pmp = length(tags), KongYiji.dict(tags)
        
        hpr, h2h, h2v, INF = fill(Tv(0), np), fill(Tv(0), (np, np)), [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        usrh2v, usrINF = [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        for sent in sents(corpus)
                pp = 0
                for (ip, iw) in sent
                        if pp == 0 hpr[ip] += 1 else h2h[pp,ip] += 1 end
                        pp = ip
                        h2v[ip][iw] = get(h2v[ip], iw, 0) + 1
                end
        end

        wmp = KongYiji.dict(words)
        
        if ispath(dictpath)
                cd(() -> begin
                                for tag in tags
                                        if !ispath(tag); continue; end
                                        ip = get(pmp, tag, 0)
                                        for file in readdir(tag)
                                                for word in eachline(joinpath(tag, file))
                                                        iw = getid!(wmp, word)
                                                        if ip > 0; usrh2v[ip][iw] = get(usrh2v[ip], iw, 0) + 1; end
                                                end
                                        end
                                end
                        end, dictpath)
        end

        dict = AhoCorasickAutomaton{Ti}(wmp)
        usr_words = length(wmp) - length(words)
        words = KongYiji.vec(wmp)
        normalize!(hpr)
        normalize!(h2h)
        normalize!(h2v, INF)
        log!(hpr)
        log!(h2h)
        normalize!(usrh2v, usrINF)
        
        HMM(dict, words, usr_words, tags, pmp, hpr, h2h, h2v, INF, usrh2v, usrINF, usrw)
end

function (hmm::HMM)(xs::Vector{String}; recover=true, withpos=false)
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(hmm.hpr)
        @assert np > 0
        dp = fill(Tv(-Inf), (nc_max + 1, np))
        pr = fill((1, 0), (nc_max + 1, np))
        return [hmm(x, dp, pr, recover, withpos) for x in xs]
end

function (hmm::HMM)(x::String; recover=true, withpos=false)
        return hmm([x]; recover=recover, withpos=withpos)[1]
end

function (hmm::HMM)(x::String, dp::Matrix{Tv}, pr::Matrix{Tuple{Int, Int}}, recover, withpos)::Vector{String}
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
                                if maybe > dp[i, pi] dp[i, pi] = maybe; pr[i, pi] = (pre_i, pj) end
                        end
                        #pre_i = i
                end
                while pv <= nv && vtxs[pv].s == i
                        vtx = vtxs[pv]
                        j = i + length(vtx)
                        for pi = 1:np
                                for pj = 1:np
                                        maybe = dp[i, pi] + hmm.h2h[pi, pj] + logph2v(hmm, pi, vtx.i)
                                        if maybe > dp[j,pj] dp[j,pj] = maybe; pr[j,pj] = (i, pi) end
                                end
                        end
                        pv = pv + 1
                end
        end
        #@show hmm
        #@show maximum(dp[nc + 1, :])
        #@show dp'
        #@show pr'
        #@show vtxs
        v = (nc + 1, argmax(dp[nc + 1,:]))
        words = fill("", 0)
        while v[1] != 1
                pv = pr[v[1],v[2]]
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
                        pv = pr[v[1],v[2]]
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

