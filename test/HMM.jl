


@testset "Generating base HMM model..." begin
        d = ChTreebank()
	m = KongYiji.HMM(d, KongYiji.dir("usrdict"), 1.)
        file = KongYiji.dir("hmm.jld2")
        save(file, "m", m)
        @time m2 = KongYiji.HMM()
        @assert m == m2

        x = KongYiji.rawsents(d)
        z = KongYiji.wordsents(d)
        y = m(x)                 
        tb = KongYiji.HmmScoreTable(z, y)
        println(tb)
        
end

#=
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

@testset "Cross validating HMM ..." begin
        @time d = ZhuXian()
        nb = 10
        batches = KongYiji.foldbatch(d, nb)
        tbs = KongYiji.HmmScoreTable[]
        @showprogress 1 "Cross Validating HMM..." for (tr, te) in batches
                hmm = KongYiji.HMM(tr, .99, KongYiji.dir("usrdict"))
                x = KongYiji.rawsents(te)
                z = KongYiji.wordsents(te)
                y = hmm(x)
                push!(tbs, KongYiji.HmmScoreTable(z, y))
                println(tbs[end])
        end
        hmmscoretable = sum(tbs)
        println(hmmscoretable)
end

@testset "Generating HMM ..." begin
        @time d = ChTreebank()
        m = HMM(d, 1., KongYiji.dir("usrdict"))
        save(KongYiji.dir("hmm.jld2"), "m", m)
        m2 = HMM()
        @test m == m2
        x = KongYiji.rawsents(d)
        z = KongYiji.wordsents(d)
        y = m(x)
        tb = KongYiji.HmmScoreTable(z, y)
        println(tb)
end

@testset "Testing HMM on icwb..." begin
        m = HMM()
        testonicwb(m)
end

@testset "Testing HMM Tagger..." begin
	#x = split("我  扔  了  两颗  手榴弹  ，  他  一下子  出  溜  下去  。")
        x = "唐山东日新能源材料有限公司"
	m = HMM()
	println(m(x; withpos=true))
end

=#
@testset "Names..." begin
        m = HMM()
        is = joinpath("D:\\", "project-names.csv")
        os = joinpath("D:\\", "project-names-segmented.csv")
        xs = [x for x in eachline(is)]
        ys = m(xs; withpos=true)
        open(os, "w") do io
        	for y in ys
        		for w in y; print(io, w, ' '); end
        		println(io)
		end
	end
end


#=

=#























