
#=
@testset "Generating base HMM model..." begin
        ctb = ChTreebank()
        hmm = KongYiji.HMM(ctb)
        file = KongYiji.dir("hmm.jld2")
        Knet.save(file, "hmm", hmm)
        zfile = KongYiji.compress(file)

        @time hmm = KongYiji.HMM()
        x = KongYiji.rawsents(ctb)
        z = KongYiji.wordsents(ctb)
        y = hmm(x)                 
        tb = KongYiji.HmmScoreTable(z, y)
        println(tb)
        
        rm(file)
end

@testset "Cross validating HMM on CTB..." begin
        @time ctb = ChTreebank()
        nb = 10
        batches = KongYiji.foldbatch(ctb, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating HMM..." for (tr, te) in batches
                hmm = KongYiji.HMM(tr)
                hmm = KongYiji.HMM(;base=hmm)
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                y = hmm(x)                 
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end

@testset "Test HMM exported debug infos..." begin
        tk = KongYiji.HMM()
        @show postable(tk)
        @show hprtable(tk)
        @show h2htable(tk)
        @show h2vtable(tk, "NT")
        @show v2htable(tk, "中国")
end
=#


@testset "Cross validating HMM on ZhuXian..." begin
        @time zx = ZhuXian()
        nb = 10
        batches = KongYiji.foldbatch(zx, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating HMM..." for (tr, te) in batches
                hmm = KongYiji.HMM(tr)
                hmm = KongYiji.HMM(;base=hmm)
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                y = hmm(x)
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end
