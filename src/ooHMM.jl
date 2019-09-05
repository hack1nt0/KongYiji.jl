
struct ooHMM
        hmm::HMM
        nn::PosLM # Neural Network based language model
end

function (hmm::ooHMM)(xs::Vector{String})
        nc_max = mapreduce(ncodeunits, max, xs)
        np = length(hmm.hmm.tags)
        dp = fill(Tv(-Inf), (np, nc_max + 1))
        pre = fill((0, 1), (np, nc_max + 1))
        hs = Matrix{Any}(undef, np, nc_max + 1)
        for ip = 1:np; hs[ip,1] = (0, 0); end
        return [hmm(x, dp, pre, hs) for x in xs]
end

function (hmm::ooHMM)(x::String)
        return hmm([x])[1]
end

function (hmm::ooHMM)(x::String, dp, pre, hs; recover=true)
        origin = x
        x = mask(x)
        chrs = codeunits(x)
        vtxs = collect(eachmatch(hmm.hmm.dict, chrs))
        sort!(vtxs)
        nv, nc, np = length(vtxs), length(chrs), size(dp, 1)
        dp .= -Inf
        #dp[:,1] .= hmm.hmm.hpr
        dp[:,1] .= 0
        reset!(hmm.nn, (0,0))
        pv = 1
        pmax = fill(1, np)
        for ic = 1:nc + 1
                while pv <= nv && vtxs[pv].s < ic; pv += 1 end
                if any(isequal(-Inf), dp[:,ic])
                        for jp = 1:np
                                jc = pmax[jp]
                                reset!(hmm.nn, hs[jp,jc])
                                posp = logsoftmax(hmm.nn([jp]))
                                for ip = 1:np
                                        maybe = dp[jp,jc] + posp[ip] + hmm.hmm.INF[jp]
                                        if dp[ip,ic] < maybe
                                                dp[ip,ic] = maybe
                                                hs[ip,ic] = deepcopy(hiddens(hmm.nn))
                                                pre[ip,ic] = (jp,jc)
                                        end
                                end
                        end
                end
                                
                for ip = 1:np
                        if dp[ip,ic] > pmax[ip]; pmax[ip] = ic; end

                        if pv > nv || ic < vtxs[pv].s; continue; end

                        if !isassigned(hs, ip, ic) 
                                @show ip, ic
                        end
                        reset!(hmm.nn, hs[ip,ic])
                        posp = logsoftmax(hmm.nn([ip]))
                        #@show exp.(posp)
                        iv = pv
                        while iv <= nv && vtxs[iv].s == ic
                                jc = ic + length(vtxs[iv])
                                for jp = 1:np
                                        maybe = dp[ip,ic] + posp[jp] + get(hmm.hmm.h2v[ip], vtxs[iv].i, hmm.hmm.INF[ip])
                                        if dp[jp,jc] < maybe
                                                dp[jp,jc] = maybe
                                                hs[jp,jc] = deepcopy(hiddens(hmm.nn))
                                                pre[jp,jc] = (ip,ic)
                                        end
                                end
                                iv += 1
                        end
                end
        end

        v = (argmax(dp[:, nc + 1]), nc + 1)
        #=
        ret = fill("", 0)
        while v[2] != 1
                pv = pre[v[1],v[2]]
                push!(ret, x[pv[2]:prevind(x, v[2])])
                v = pv
        end
        reverse!(ret)
        if recover ret = demask(ret, origin) end
        =#
        ret = fill(("",""), 0)
        while v[2] != 1
                pv = pre[v[1],v[2]]
                push!(ret, (hmm.hmm.tags[pv[1]], x[pv[2]:prevind(x, v[2])]))
                v = pv
        end
        reverse!(ret)
        return ret
end

#### Utils

function ==(a::ooHMM, b::ooHMM)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(ooHMM))
end
