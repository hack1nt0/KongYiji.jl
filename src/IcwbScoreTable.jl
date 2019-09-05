
struct IcwbScoreTable
        TOTAL_INSERTIONS
        TOTAL_DELETIONS
        TOTAL_SUBSTITUTIONS
        TOTAL_NCHANGE
        TOTAL_TRUE_WORD_COUNT
        TOTAL_TEST_WORD_COUNT
        TOTAL_TRUE_WORDS_RECALL
        TOTAL_TEST_WORDS_PRECISION
        F_MEASURE
        OOV_Rate
        OOV_Recall_Rate
        IV_Recall_Rate

        n::Int
        source::String
###	D:\KongYiji.jl\src\KongYiji.jl\..\..\test\as_kongyiji_tokens.txt	32323	2437	38777	73537	122610	152496	0.664	0.534	0.592	0.043	0.113	0.689

end

function show(io::IO, tb::IcwbScoreTable)
        fnames = fieldnames(IcwbScoreTable)
        nf = length(fnames)
        mat = Matrix{Any}(undef, (nf - 2, 2))
        for i = 1:nf - 2
                val = getfield(tb, fnames[i]) / tb.n
                val = i <= 6 ? Int(round(val)) : trunc(val; digits=3)
                mat[i,:] = [fnames[i], val]
        end
        show(io, UselessTable(mat; heads=["icwb $(tb.source) score table $(tb.n) combined"], cnames=["measure", "score"]))
end

+(a::IcwbScoreTable, b::IcwbScoreTable) = IcwbScoreTable(map(x->getfield(a, x) + getfield(b, x), fieldnames(IcwbScoreTable)[:end-1])..., a.source)

function IcwbScoreTable(dict::Set{String}, truths::Vector{Vector{SubString{String}}}, outputs::Vector{Vector{String}}, source::String)
        n = length(truths)
        maxl = 2000 #todo
        dp = Matrix{Int}(undef, (maxl, maxl))
        prev = Matrix{Tuple{Int, Int}}(undef, (maxl, maxl))
        nins, ndel, nsub, ntruew, ntestw, nhits, noov, oov_hits, niv, iv_hits = repeat([0], 10)
        for (a, b) in zip(truths, outputs)
                na, nb = length(a), length(b)
                for i in 1:na, j in 1:nb
                        dp[i,j] = 0
                        if a[i] == b[j] 
                                dp[i,j] = get(dp, (i - 1,j - 1), 0) + 1 
                                prev[i,j] = (i - 1,j - 1)
                        end
                        if dp[i,j] < get(dp, (i - 1,j), 0)
                                dp[i,j] = get(dp, (i - 1,j), 0)
                                prev[i,j] = get(prev, (i - 1,j), (0,0))
                        end
                        if dp[i,j] < get(dp, (i,j - 1), 0)
                                dp[i,j] = get(dp, (i,j - 1), 0)
                                prev[i,j] = get(prev, (i,j - 1), (0,0))
                        end
                end
                vs = [(max(na, nb), max(na, nb)), (na, nb)]
                ia, ib = na, nb
                while true
                        ia, ib = vs[end]
                        if ia == 0 || ib == 0 break end
                        push!(vs, prev[ia,ib])
                end
                push!(vs, (0,0))
                reverse!(vs)
                for i = 2:length(vs)
                        da, db = vs[i] .- vs[i - 1]
                        if da == db nsub += da - 1
                        elseif da < db nins += db - da; nsub += db - da
                        else ndel += da - db; nsub += da - db
                        end
                end
                ntruew += na
                ntestw += nb
                nhits += dp[na,nb]
                for wa in a !in(wa, dict) ? noov += 1 : niv += 1 end
                for v in vs[2:end-1] all(v .> 0) && !in(b[v[2]], dict) ? oov_hits += 1 : iv_hits += 1 end
        end
        recall, precision = nhits / ntruew, nhits / ntestw
        F1 = f1(precision, recall)
        oov_rate = noov / ntruew
        oov_recall = oov_hits / noov
        iv_recall = iv_hits / niv
        return IcwbScoreTable(nins, ndel, nsub, (nins+ndel+nsub), ntruew, ntestw, recall, precision, F1, oov_rate, oov_recall, iv_recall, 1, source)
end

