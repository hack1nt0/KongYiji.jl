
mutable struct HMM
    dict::AhoCorasickAutomaton{Ti}
    words::Vector{String}
    usr_words::Int
    tags::Vector{String}
    hpr::Vector{Tv}
    h2h::Matrix{Tv}
    h2v::Vector{Dict{Int, Tv}}
    INF::Vector{Tv}
end

function HMM(corpus, dictpath, major)
        ispath(dictpath) || @warn "dictpath is not a valid path"
        @assert 0. < major <= 1.

        tags, words = KongYiji.postags(corpus), KongYiji.words(corpus)
        np = length(tags)
        
        hpr, h2h, h2v, INF = fill(Tv(0), np), fill(Tv(0), (np, np)), [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        h2v2 = [Dict{Int, Tv}() for _ in 1:np]

        for sent in sents(corpus)
                pp = 0
                for (ip, iw) in sent
                        if pp == 0 hpr[ip] += 1 else h2h[pp,ip] += 1 end
                        pp = ip
                        h2v[ip][iw] = get(h2v[ip], iw, 0) + 1
                end
        end

        if major < 1.
                  h2v2 = [Dict{Int, Tv}() for _ in 1:np]
                  for ip = 1:np
                          pct, tot = 0., sum(values(h2v[ip]))
                          for (iw, cnt) in sort!(collect(h2v[ip]); by=last, rev=true)
                                  h2v2[ip][iw] = cnt
                                  pct += cnt / tot
                                  if pct >= major; break; end
                          end
                  end
                  h2v = h2v2
        end


        for ip = 1:np
                if !startswith(tags[ip], "N"); continue; end
                for (iw, cnt) in h2v[ip]; h2v[ip][iw] = 1; end
        end
        

        wmp = KongYiji.dict(words)
        
        if isdir(dictpath)
                cd(() -> begin
                                for tag in tags
                                        if !ispath(tag); continue; end
                                        pmp = KongYiji.dict(tags)
                                        ip = get(pmp, tag, 0)
                                        if ip == 0; continue; end
                                        maxcnt = maximum(values(h2v[ip]))
                                        for file in readdir(tag)
                                                for word in eachline(joinpath(tag, file))
                                                        word = mask(word) #todo forgotten
                                                        iw = getid!(wmp, word)
                                                        if ip > 0
                                                                if startswith(tag, "N"); h2v[ip][iw] = 1;
                                                                else h2v[ip][iw] = maxip;
                                                                end
                                                        end
                                                end
                                        end
                                end
                        end, dictpath)
        end

        dict = AhoCorasickAutomaton{Ti}(wmp)
        usr_words = length(wmp) - length(words)
        words = KongYiji.vec(wmp)
        normalize!.((hpr, h2h))
        normalize!(h2v, INF)
        log!.((hpr, h2h, h2v, INF))

        HMM(dict, words, usr_words, tags, hpr, h2h, h2v, INF)
end


HMM() = load(KongYiji.dir("hmm.jld2"))["m"]

function (m::HMM)(xs::Vector{String}; recover=true, withpos=false)
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(m.hpr)
        @assert np > 0
        dp = fill(Tv(-Inf), (nc_max + 1, np))
        pr = fill((1, 0), (nc_max + 1, np))
        return [m(x, dp, pr, recover, withpos) for x in xs]
end

function (m::HMM)(x::String; recover=true, withpos=false)
        return m([x]; recover=recover, withpos=withpos)[1]
end

function (m::HMM)(x::String, dp::Matrix{Tv}, pr::Matrix{Tuple{Int, Int}}, recover, withpos)::Vector{String}
        origin = x
        x = mask(x)
        chrs = codeunits(x) #todo slow?
        vtxs = collect(eachmatch(m.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length.((vtxs, chrs, m.hpr))
        for i = 1:nc + 1, j in 1:np dp[i,j] = -Inf end
        pv = 1
        dp[1, :] = m.hpr
        pre_i = 1
        for i in 1:nc + 1
                if dp[i,1] != -Inf pre_i = i end
                while pv <= nv && vtxs[pv].s < i pv = pv + 1 end
                if !(pv <= nv && vtxs[pv].s == i || i == nc + 1) continue end
                if dp[i, 1] == -Inf
                        #@show i, pre_i
                        for pi = 1:np, pj = 1:np
                                maybe = dp[pre_i, pj] + m.h2h[pj, pi] + logph2v(m, pj, 0)
                                if maybe > dp[i, pi] dp[i, pi] = maybe; pr[i, pi] = (pre_i, pj) end
                        end
                        #pre_i = i
                end
                while pv <= nv && vtxs[pv].s == i
                        vtx = vtxs[pv]
                        j = i + length(vtx)
                        for pi = 1:np
                                for pj = 1:np
                                        maybe = dp[i, pi] + m.h2h[pi, pj] + logph2v(m, pi, vtx.i)
                                        if maybe > dp[j,pj] dp[j,pj] = maybe; pr[j,pj] = (i, pi) end
                                end
                        end
                        pv = pv + 1
                end
        end
        #@show m
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
                        postag = m.tags[pv[2]]
                        push!(postags, postag)
                        v = pv
                end
                reverse!(postags)
                ret = [join(p, "/") for p in zip(words, postags)]
        end

        return ret
end

logph2v(m::HMM, p, v) = haskey(m.h2v[p], v) ? m.h2v[p][v] : m.INF[p]

function ==(a::HMM, b::HMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(HMM))
end

# tagger
function (m::HMM)(xs::Vector{Vector{T}}; recover=true, withword=true) where {T}
	maxnw = mapreduce(length, max, xs)
	np = length(m.tags)
	dp = fill(-oo, (np, maxnw + 1))
	pr = fill((0, 1), (np, maxnw + 1))
	[m(x, dp, pr, recover, withword) for x in xs]
end

function (m::HMM)(x::Vector{T}; recover=true, withword=false) where {T}
        return m([x]; recover=recover, withword=withword)[1]
end

function (m::HMM)(x::Vector{T}, dp::Matrix{Tv}, pr::Matrix{Tuple{Int, Int}}, recover, withword)::Vector{String} where {T}
	nw, np = length.((x, m.hpr))
        for i = 1:nw + 1, j in 1:np dp[j, i] = -oo end
        dp[:, 1] = m.hpr

        for i = 1:nw
        	iw = get(m.dict, mask(x[i]), Ti(0))
        	j = i + 1
        	for ip = 1:np, jp = 1:np
        		maybe = dp[ip, i] + get(m.h2v[ip], iw, m.INF[ip]) + m.h2h[ip, jp]
        		if dp[jp, j] < maybe; dp[jp, j] = maybe; pr[jp, j] = (ip, i); end
		end
	end

	v = (argmax(dp[:, nw + 1]), nw + 1)
	r = fill("", 0)
        while v[2] != 1
                pv = pr[v[1],v[2]]
                postag = m.tags[pv[1]]
                word = x[pv[2]]
                if !recover; word = mask(word); end
                push!(r, string(word, '/', postag))
                v = pv
        end
        reverse!(r)
        r
end


