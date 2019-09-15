
struct NGramHMM
        grams::Int
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

function NGramHMM(grams::Int, corpus, dictpath::String, usrw::Float64)
        @assert grams >= 2
        ispath(dictpath) || @warn "dictpath is not a valid path"
        @assert usrw < 1.

        tags, words = KongYiji.postags(corpus), KongYiji.words(corpus)
        np, pmp = length(tags), KongYiji.dict(tags)
        B = np + 1
        nf = (B ^ (grams - 1)) - 1        
        hpr, h2h, h2v, INF = fill(Tv(0), np), fill(Tv(0), (nf, np)), [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        usrh2v, usrINF = [Dict{Int, Tv}() for _ in 1:np], fill(Tv(0), np)
        for sent in sents(corpus)
                ns = length(sent)
                for i = 1:ns
                        ip, iw = sent[i]
                        if i == 1; hpr[ip] += 1;
                        else
                                from, pow = 0, 1
                                for j = i - 1:-1:max(1, i - grams + 1); from += sent[j][1] * pow; pow *= B; end
                                h2h[from, ip] += 1
                        end
                        h2v[ip][iw] = get(h2v[ip], iw, 0) + 1
                end
        end
        

        wmp = KongYiji.dict(words)
        
        if ispath(dictpath)
                cd(() -> begin
                                #@show pwd()
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
        
        NGramHMM(grams, dict, words, usr_words, tags, pmp, hpr, h2h, h2v, INF, usrh2v, usrINF, usrw)
end

NGramHMM() = load(KongYiji.dir("ngramhmm.jld2"))["m"]

function (m::NGramHMM)(xs::Vector{String}; recover=true, withpos=false)
        nc_max = mapreduce(ncodeunits, max, xs)
        nf, np = size(m.h2h)
        dp = fill(Tv(-Inf), (nf, nc_max + 1))
        pr = fill((0, 1), (nf, nc_max + 1))
        ma = fill(false, nc_max + 1)
        return [m(x, dp, pr, ma, recover, withpos) for x in xs]
end

function (m::NGramHMM)(x::String; o...)
        return m([x]; o...)[1]
end

function (m::NGramHMM)(x::String, dp::Matrix{Tv}, pr::Matrix{Tuple{Int,Int}}, ma::Vector{Bool}, recover::Bool, withpos::Bool)::Vector{String} where {T}
        
        origin = x
        x = mask(x)
        chrs = codeunits(x) #todo slow?
        vtxs = collect(eachmatch(m.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length.((vtxs, chrs, m.hpr))
        for i = 1:nc + 1; dp[:, i] .= -oo end
        pv = 1
        dp[1:np, 1] .= m.hpr
        prei = 1
        B, M = np + 1, (np + 1) ^ (m.grams - 2)
        ma[1] = true; ma[2:nc + 1] .= false;
        nf = size(m.h2h, 1)
        
        for i in 1:nc + 1

                if ma[i]; prei = i; end

                while pv <= nv && vtxs[pv].s < i; pv = pv + 1; end

                go = i == nc + 1 || pv <= nv && vtxs[pv].s == i
                if !go; continue; end

                if !ma[i]
                        j = prei
                        fromL, fromR = 1, np
                        for g = 1:m.grams - 1
                                if dp[fromL, j] != -oo
                                        for from = fromL:fromR
                                                jp = from % B
                                                if jp == 0; continue; end
                                                for ip = 1:np
                                                        to = from % M * B + ip
                                                        maybe = dp[from, j] + m.h2h[from, ip] + logph2v(m, jp, 0)
                                                        if dp[to, i] < maybe
                                                                dp[to, i] = maybe
                                                                pr[to, i] = (from, j)
                                                        end
                                                end
                                        end
                                end
                                fromL = fromL * B + 1
                                fromR = fromR * B + np
                        end
                        #prei = i
                        ma[i] = true
                end

                while pv <= nv && vtxs[pv].s == i
                        vtx = vtxs[pv]
                        j = i + length(vtx)
                        ma[j] = true
                        fromL, fromR = 1, np
                        for g = 1:m.grams - 1
                                if dp[fromL, i] != -oo
                                        for from = fromL:fromR
                                                 ip = from % B
                                                 if ip == 0; continue; end
                                                 for jp = 1:np
                                                         to = from % M * B + jp
                                                         #if ip == 0; @show from, B, fromL, fromR, g; end
                                                         maybe = dp[from, i] + m.h2h[from, jp] + logph2v(m, ip, vtx.i)
                                                         if dp[to, j] < maybe
                                                                 dp[to, j] = maybe
                                                                 pr[to, j] = (from, i)
                                                         end
                                                 end
                                        end
                                end
                                #=
                                else
                                        @assert all(isequal(-oo), dp[fromL:fromR, i])
                                end
                                =#
                                fromL = fromL * B + 1
                                fromR = fromR * B + np
                        end
                        pv = pv + 1
                end
        end
        
        #@show m.h2h
        #@show maximum(dp[:, nc + 1])
        #@show dp
        #@show pr
        #@show vtxs
        v = (argmax(dp[:, nc + 1]), nc + 1)
        words = fill("", 0)
        while v[2] != 1
                pv = pr[v...]
                word = x[pv[2]:prevind(x, v[2])]
                push!(words, word)
                v = pv
        end
        reverse!(words)
        if recover; words = demask(words, origin); end
        ret = words

        if withpos
                v = (argmax(dp[:, nc + 1]), nc + 1)
                postags = fill("", 0)
                while v[2] != 1
                        pv = pr[v...]
                        postag = m.tags[v[1] % np]
                        push!(postags, postag)
                        v = pv
                end
                reverse!(postags)
                ret = [join(p, "/") for p in zip(words, postags)]
        end
        
        #@show origin
        #@show ret
        return ret
end

logph2v(m::NGramHMM, p, v) = ((1. - m.usrw) * get(m.h2v[p], v, m.INF[p]) + m.usrw * get(m.usrh2v[p], v, m.usrINF[p])) |> log

function ==(a::NGramHMM, b::NGramHMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(NGramHMM))
end
