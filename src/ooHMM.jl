

struct ooHMM
        hmm::HMM
        lm::LM # RNN based language model
end

import Base.+
+(hmm::HMM, lm::LM) = ooHMM(hmm, lm)

ooHMM() = Knet.load(KongYiji.load("oohmm.jld2"), "oohmm")

function (hmm::ooHMM)(xs::Vector{String}; o...)
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(hmm.hmm.tags)
        dp = fill(Tv(-oo), (np, nc_max + 1))
        pre = fill((0, 1), (np, nc_max + 1))
        hs = Matrix{Any}(undef, np, nc_max + 1)
        for ip = 1:np; hs[ip,1] = (0, 0); end
        return [hmm(x, dp, pre, hs; o...) for x in xs]
end

function (hmm::ooHMM)(x::String; o...)
        return hmm([x]; o...)[1]
end

function (hmm::ooHMM)(x::String, dp, pre, hs; recover=true, withpos=false)
        #@show x
        origin = x
        x = mask(x)
        chrs = codeunits(x)
        vtxs = collect(eachmatch(hmm.hmm.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length(vtxs), length(chrs), size(dp, 1)
        dp .= -oo
        dp[:,1] .= hmm.hmm.hpr
        #dp[:,1] .= 0
        reset!(hmm.lm, (0,0))
        pv = 1
        preic = 0
        for ic = 1:nc + 1
                while pv <= nv && vtxs[pv].s < ic; pv += 1 end

                if dp[1,ic] != -oo
                        preic = ic
                elseif ic == nc + 1 || pv <= nv && vtxs[pv].s == ic
                        jc = preic
                        for jp = 1:np
                                reset!(hmm.lm, hs[jp,jc])
                                posp = logsoftmax(hmm.lm([jp]))
                                for ip = 1:np
                                        maybe = dp[jp,jc] + posp[ip] + logph2v(hmm.hmm, jp, 0)
                                        if dp[ip,ic] < maybe
                                                dp[ip,ic] = maybe
                                                hs[ip,ic] = deepcopy(hiddens(hmm.lm))
                                                pre[ip,ic] = (jp,jc)
                                        end
                                end
                        end
                end

                if dp[1,ic] == -oo || ic == nc + 1; continue; end

                for ip = 1:np
                        #=
                        if !isassigned(hs, ip, ic) 
                                @show ip, ic
                        end
                        =#
                        reset!(hmm.lm, hs[ip,ic])
                        posp = logsoftmax(hmm.lm([ip]))
                        #@show exp.(posp)
                        iv = pv
                        while iv <= nv && vtxs[iv].s == ic
                                jc = ic + length(vtxs[iv])
                                for jp = 1:np
                                        maybe = dp[ip,ic] + posp[jp] + logph2v(hmm.hmm, ip, vtxs[iv].i)
                                        if dp[jp,jc] < maybe
                                                dp[jp,jc] = maybe
                                                hs[jp,jc] = deepcopy(hiddens(hmm.lm))
                                                pre[jp,jc] = (ip,ic)
                                        end
                                end
                                iv += 1
                        end
                end
        end

        v = (argmax(dp[:,nc + 1]), nc + 1)
        words = String[]
        while v[2] != 1
                pv = pre[v[1],v[2]]
                word = x[pv[2]:prevind(x, v[2])]
                push!(words, word)
                v = pv
        end
        reverse!(words)
        if recover; words = demask(words, origin); end
        ret = words

        if withpos
                v = (argmax(dp[:,nc + 1]), nc + 1)
                postags = String[]
                while v[2] != 1
                        pv = pre[v[1],v[2]]
                        postag = hmm.tags[pv[1]]
                        push!(postags, postag)
                        v = pv
                end
                reverse!(postags)
                ret = [join(p, "/") for p in zip(words, postags)]
        end

        return ret
end

#### Utils

function ==(a::ooHMM, b::ooHMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(ooHMM))
end

