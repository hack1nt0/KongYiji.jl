

@testset "Generating CTB data file..." begin
        home = joinpath("d:\\", "ctb8.0")
        ctb = KongYiji.ChTreebank(home; nf=0)
        @show ctb
        ctb = ctb |> KongYiji.mask
        @show ctb

        file = KongYiji.dir("ctb.jld2")
        @time Knet.save(file, "ctb", ctb)
        @time zfile = KongYiji.compress(file)
        rm(file)
        @time file = KongYiji.decompress(zfile)
        @time ctb2 = Knet.load(file, "ctb")
        @test ctb == ctb2
end

