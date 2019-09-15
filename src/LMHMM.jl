

struct LMHMM
        hmm::HMM
        lm::LM # RNN based language model
end

import Base.+
+(hmm::HMM, lm::LM) = LMHMM(hmm, lm)

LMHMM() = Knet.load(KongYiji.load("lmhmm.jld2"), "lmhmm")

struct Ts
        val::Tv
        ic::Int
        ip::Int
        bp::Union{Ts, Nothing}
        hc
end

import Base.isless
isless(a::Ts, b::Ts) = a.val < b.val

function (m::LMHMM)(xs::Vector{String}; beams=100, o...)
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(m.hmm.tags)
        beams = max(beams, np)
        #Th = Heap{Ts{typeof(hiddens(m.lm))}}
        Th = Heap{Ts}

        dp = Vector{Th}(undef, nc_max + 1)
        for i = 1:nc_max + 1; dp[i] = Th(beams); end
        for i = 1:np; push!(dp[1], Ts(0, 1, i, nothing, (0,0))); end

        return [m(x, dp; o...) for x in xs]
end

function (m::LMHMM)(x::String; o...)
        return m([x]; o...)[1]
end

function (m::LMHMM)(x::String, dp::T; recover=true, withpos=false) where T
        @show x
        origin = x
        x = mask(x)
        chrs = codeunits(x)
        vtxs = collect(eachmatch(m.hmm.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length(vtxs), length(chrs), size(dp, 1)
        pv = 1
        preic = 0
        for i = 2:nc + 1; empty!(dp[i]); end
        for ic = 1:nc + 1
                #@show ic, length(dp[ic])
                while pv <= nv && vtxs[pv].s < ic; pv += 1 end
                
                if !isempty(dp[ic])
                        preic = ic
                elseif (ic == nc + 1 || pv <= nv && vtxs[pv].s == ic)
                        jc = preic
                        for js in dp[jc]
                                is = next(m, js, 0, ic)
                                push!(dp[ic], is)
                        end
                        #empty!(dp[preic]) #TODO
                end

                if pv > nv || ic < vtxs[pv].s || ic == nc + 1; continue; end

                while pv <= nv && vtxs[pv].s == ic
                        jc = ic + length(vtxs[pv])
                        for is in dp[ic]
                                js = next(m, is, vtxs[pv].i, jc)
                                push!(dp[jc], js)
                        end
                        pv += 1
                end
        end

        t = findmax(dp[nc + 1].h.valtree) |> first
        cur = t
        words = String[]
        while !isnothing(cur.bp)
                pre = cur.bp
                push!(words, x[pre.ic:prevind(x, cur.ic)])
                cur = pre
        end
        reverse!(words)
        if recover; words = demask(words, origin); end
        ret = words

        if withpos
                cur = t
                postags = String[]
                while !isnothing(cur.bp)
                        pre = cur.bp
                        push!(postags, m.hmm.postags[cur.ip])
                        cur = pre
                end
                reverse!(postags)
                ret = [join(p, '/') for p in zip(words, postags)]
        end

        return ret
end

function next(m::LMHMM, s::Ts, iw::Int, ic::Int)
        val = logph2v(m.hmm, s.ip, iw)
        reset!(m.lm, s.hc)
        #@show typeof(s.hc)
        probp = m.lm(s.ip) |> vec
        #@show typeof(probp)
        ip = findmax(probp) |> last
        #@show typeof(ip)

        hc = hiddens(m.lm) |> deepcopy
        Ts(val, ic, ip, s, hc)
end


#### Utils

function ==(a::LMHMM, b::LMHMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(LMHMM))
end

