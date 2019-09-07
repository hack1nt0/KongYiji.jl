
@testset "Language Model..." begin
        @time d = ZhuXian()
        lm = LM(d)
        #Knet.save(KongYiji.dir("lm.jld2"), "zx", lm)
        #r = generate!(lm, 1; words=d.postags, len=100)
        #println(r)
end
               