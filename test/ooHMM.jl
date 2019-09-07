
@testset "Cross validating ooHMM on CTB..." begin
        @time ctb = ChTreebank()
        #@time ctb = ZhuXian()

        nb = 10
        batches = KongYiji.foldbatch(ctb, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating ooHMM..." for (tr, te) in batches
                base = KongYiji.HMM(tr)
                hmm = KongYiji.HMM(;base=base)
                lm = LM(tr; rnnType=:lstm, numLayers=1, bidirectional=false)
                oohmm = hmm + lm
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                @time y = oohmm(x; recover=false, withpos=false)                 
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end
