
@testset "Generating ZhuXian data file..." begin
        home = "D:\\multi-criteria-cws\\data\\other\\zx"
        zx = ZhuXian(home)
        @show zx

        file = KongYiji.dir("zhuxian.jld2")
        @time Knet.save(file, "zx", zx)
        @time zfile = KongYiji.compress(file)
        rm(file)
        @time file = KongYiji.decompress(zfile)
        @time zx2 = Knet.load(file, "zx")
        @test zx == zx2
end

