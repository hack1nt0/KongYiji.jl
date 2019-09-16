

@testset "Cross validating LMHMM on ZX..." begin
        @time ctb = ZhuXian()

        nb = 10
        batches = KongYiji.foldbatch(ctb, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating ooHMM..." for (tr, te) in batches
                hmm = KongYiji.HMM(tr, "", 0.)
                lm = KongYiji.LM(tr; epochs=20, rnnType=:lstm, numLayers=1, bidirectional=false)
                m = hmm + lm
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                @time y = m(x; beams=500, recover=false, withpos=false)
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end
