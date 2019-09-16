

#=
@testset "test NGramHMM..." begin
        @time d = ZhuXian()
        @time m = KongYiji.NGramHMM(2, d, KongYiji.dir("usrdict1"), 0.)
        @time m2 = KongYiji.HMM(d, KongYiji.dir("usrdict1"), 0.)
        @test m.hpr == m2.hpr
        @test m.h2h == m2.h2h
        @test m.h2v == m2.h2v
        @test m.INF == m2.INF


        xs = "田不易独自一人站在树林里的僻静处，负手而立。"
        r = m(xs)
        r2 = m2(xs)
        println(r)
        println(r2)
end
  =#


@testset "Cross validating NGramHMM..." begin
        @time ctb = ChTreebank()

        nb = 10
        batches = KongYiji.foldbatch(ctb, nb)
        tbs2 = KongYiji.HmmScoreTable[]
        tbs3 = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating ooHMM..." for (tr, te) in batches
                m3 = KongYiji.NGramHMM(3, tr, KongYiji.dir("usrdict1"), 0.)
                m2 = KongYiji.HMM(tr, KongYiji.dir("usrdict1"), 0.)
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                @time y2 = m2(x)
                @time y3 = m3(x)
                t2 = KongYiji.HmmScoreTable(z, y2)
                t3 = KongYiji.HmmScoreTable(z, y3)
                @show t2
                @show t3
                push!(tbs2, t2)
                push!(tbs3, t3)
                #println(tbs[end])
        end
        println(sum(tbs2))
        println(sum(tbs3))
end

#=
@testset "Compare NGramHMM with HMM..." begin
        @time d = ZhuXian()
        @time m = KongYiji.NGramHMM(2, d, KongYiji.dir("usrdict1"), 0.)
        @time m2 = KongYiji.HMM(d, KongYiji.dir("usrdict1"), 0.)
        for sent in rawsents(d)
                y = m(sent)
                y2 = m2(sent)
                @test y == y2
                if y != y2; println(sent); end
        end
end

  =#