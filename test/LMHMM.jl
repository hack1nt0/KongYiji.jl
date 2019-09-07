

@testset "Cross validating LMHMM on ZX..." begin
        @time ctb = ZhuXian()

        nb = 10
        batches = KongYiji.foldbatch(ctb, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating ooHMM..." for (tr, te) in batches
                base = KongYiji.HMM(tr)
                hmm = KongYiji.HMM(;base=base)
                lm = LM(tr; rnnType=:lstm, numLayers=1, bidirectional=false)
                lmhmm = hmm + lm
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                @time y = oohmm(x; recover=false, withpos=false)                 
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end
