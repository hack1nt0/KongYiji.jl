using Test, KongYiji, Pkg, JLD2, FileIO, ProgressMeter, DataFrames, Knet


#include("ChTreebank.jl")
#include("HMM.jl")
#include("ZhuXian.jl")
include("LM.jl")
#include("ooHMM.jl")
#include("LMHMM.jl")













#=
@testset "Test hmm-lstm..." begin
        #KongYiji.trhmmlstm()
        ctb = ChTreebank()
        #ctb = [[[("pa", "wa")]]]
        m = HMM2(ctb; embedsz=64, hiddensz=64, epochs=1, batchsz=256, seqlen=100, file=joinpath(pathof(KongYiji), "..", "..", "data", "hmm2.jld2"))
        tks = m("一个脱离了低级趣味的人")
        @show tks
end
=#

#=
@testset "Extract wrong cases on icwb..." begin
        home = joinpath("d:\\", "icwb2-data")
        tk = Kong()
        @showprogress 1 "" for source in ["pku", "msr", "cityu", "as"]
                test_file = joinpath(home, "testing", "$(source)_test.utf8")
                truth_file = joinpath(home, "gold", source == "as" ? "as_testing_gold.utf8" : "$(source)_test_gold.utf8")
                dict_file = joinpath(home, "gold", "$(source)_training_words.utf8")
                #tk = Kong(; user_dict_path=dict_file)
                output = tk([line for line in eachline(test_file) if length(line) > 0])
                truth = [split(line) for line in eachline(truth_file) if length(line) > 0]
                debug_file = joinpath(pathof(KongYiji), "..", "..", "test", "wrong.cases.$(source).2.txt")
                open(debug_file, "w") do io
                        for i in 1:length(truth)
                                o, s = output[i], truth[i]
                                ms = eachmatch(o, s)
                                for i = 1:length(ms) - 1
                                        lo, ls = ms[i] .+ 1
                                        ro, rs = ms[i + 1] .- 1
                                        if lo <= ro
                                                print(io, "S : "); for j in ls:rs print(io, s[j], '\t') end; println(io)
                                                print(io, "O : "); for j in lo:ro print(io, o[j], '\t') end; println(io)
                                                println(io, "-------------------------")
                                        end
                                end
                                println(io, "=================================")
                        end
                end
        end
end
=#

#=
@testset "Do statistics on ctb.." begin
        @time ctb_home = KongYiji.unzip7(joinpath(pathof(KongYiji), "..", "..", "data", "ctb.jld2.7z"))
        @time ctb = load(ctb_home)["ctb"]
        df = DataFrame(cw=String[], cpos=String[], lw=String[], rw=String[], lpos=String[], rpos=String[])
        for doc in ctb, sent in doc
                nw = length(sent)
                for i in 1:nw
                        cpos, cw = sent[i]
                        if word == cw | pos == cpos
                                #l = i == 1  ? "^" : sent[i - 1][2][end:end]
                                #r = i == nw ? "\$" : sent[i + 1][2][end:end]
                                lpos, lw = i == 1  ? ("^", "") : sent[i - 1]
                                rpos, rw = i == nw ? ("\$", "") : sent[i + 1]
                                push!(df, (cw, cpos, lw, rw, lpos, rpos))
                        end
                end
        end
        println(size(df))
end
=#

#=
@testset "Testing on icwb..." begin
        home = joinpath("d:\\", "icwb2-data")
        @showprogress 1 "" for source in ["pku", "msr", "cityu", "as"]
                test_file = joinpath(home, "testing", "$(source)_test.utf8")
                truth_file = joinpath(home, "gold", source == "as" ? "as_testing_gold.utf8" : "$(source)_test_gold.utf8")
                dict_file = joinpath(home, "gold", "$(source)_training_words.utf8")
                tk = Kong(; user_dict_path=dict_file)
                output = tk([line for line in eachline(test_file) if length(line) > 0])
                truth = [split(line) for line in eachline(truth_file) if length(line) > 0]
                debug_file = joinpath(pathof(KongYiji), "..", "..", "test", "debug.$(source).txt")
                open(debug_file, "w") do io
                       for i in 1:length(truth)
                               print(io, "S : ")
                               for w in truth[i] print(io, w, "\t") end
                               println(io)
                               print(io, "O : ")
                               for w in output[i] print(io, w, "\t") end
                               println(io)
                               println(io)
                       end
                end
                #=
                dict = Set([word for word in eachline(dict_file) if length(word) > 0])
                println(KongYiji.IcwbScoreTable(dict, truth, output, source))
                println("==================================")
                =#
        end
end
=#

#=
@testset "Testing on icwb..." begin
        home = joinpath("d:\\", "icwb2-data")
        scorer = joinpath(home, "scripts", "score.pl")
        tk = Kong()
        for source in ["as"] #["pku", "msr", "cityu", "as"]
                test = joinpath(home, "testing", "$(source)_test.utf8")
                test_out = joinpath(pathof(KongYiji), "..", "..", "test", "$(source)_kongyiji_tokens.txt")
                open(test_out, "w") do io
                        for line in tk(collect(eachline(test)))
                                for token in line print(io, token, ' ') end
                                println(io)
                        end
                end
                dict = joinpath(home, "gold", "$(source)_training_words.utf8")
                truth = joinpath(home, "gold", source == "as" ? "as_testing_gold.utf8" : "$(source)_test_gold.utf8")
                cmd = "perl -w $(scorer) $(dict) $(truth) $(test_out)"
                run(pipeline(ifelse(Sys.iswindows(), `cmd /c $cmd`, `sh -c $cmd`); 
                                stdout=joinpath(pathof(KongYiji), "..", "..", "test", "$(source)_kongyiji_score.txt")))
        end
end
=#

#=
@testset "Generating REQUIRE file..." begin
        println(Pkg.METADATA_compatible_uuid("KongYiji"))
        PT = Pkg.Types
        Pkg.activate("..")             # current directory as the project
        ctx = PT.Context()
        pkg = ctx.env.pkg
        if pkg ≡ nothing
            @error "Not in a package, I won't generate REQUIRE."
            exit(1)
        else
            @info "found package" pkg = pkg
        end

        deps = PT.get_deps(ctx)
        non_std_deps = sort(collect(setdiff(keys(deps), values(ctx.stdlibs))))

        open(joinpath("..", "REQUIRE"), "w") do io
            println(io, "julia 0.7")
            for d in non_std_deps
                println(io, d)
                @info "listing $d"
            end
        end
end
=#

#=
@testset "Generating HMM model file of CTB..." begin
        @time ctb_home = KongYiji.unzip7(joinpath(pathof(KongYiji), "..", "..", "data", "ctb.jld2.7z"))
        @time ctb = load(ctb_home)["ctb"]
        @time hmm = KongYiji.HMM(ctb)
        home = joinpath(pathof(KongYiji), "..", "..", "data", "hmm.jld2")
        mkpath(dirname(home))
        @time @save home hmm
        @time zhome = KongYiji.zip7(home)
        rm(home)
        @time home2 = KongYiji.unzip7(zhome)
        @assert home == home2
        @time hmm2 = load(home2)["hmm"]
        @test hmm == hmm2
end
=#

#=
@testset "Test KongYiji(1) with Hand written examples..." begin
        tk = Kong()
        input = "一个脱离了低级趣味的人"
        output = tk(input)
        @show output

        input = "一/个/脱离/了/低级/趣味/的/人"
        tk(input, "/")

        inputs = [
                "他/说/的/确实/在理",
                "这/事/的确/定/不/下来",
                "费孝通/向/人大/常委会/提交/书面/报告",
                "邓颖超/生前/使用/过/的/物品",
                "停电/范围/包括/沙坪坝区/的/犀牛屙屎/和/犀牛屙屎抽水",
        ]
        println("Input :")
        for input in inputs
                println(input)
        end

        println("raw output :")
        for input in inputs
                println(tk(filter(c -> c != '/', input)))
        end
        
        tk2 = Kong(; user_dict_array=[("VV", "定"),
                                      ("VA", "在理"),
                                       "邓颖超",
                                       "沙坪坝区", 
                                       "犀牛屙屎",
                                       "犀牛屙屎抽水",
                                     ]
        )
        println("output with user dict supplied :")
        for input in inputs
                println(tk2(filter(c -> c != '/', input)))
        end
end

=#