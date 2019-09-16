

function testonicwb(m; home=joinpath("d:\\", "icwb2-data"))
        for source in ["pku", "msr", "cityu", "as"]
                test_file = joinpath(home, "testing", "$(source)_test.utf8")
                truth_file = joinpath(home, "gold", source == "as" ? "as_testing_gold.utf8" : "$(source)_test_gold.utf8")
                dict_file = joinpath(home, "gold", "$(source)_training_words.utf8")
                x = [line for line in eachline(test_file) if length(line) > 0]
                z = [split(line) for line in eachline(truth_file) if length(line) > 0]
                y = m(x)
                #=
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
                =#
                dict = Set([word for word in eachline(dict_file) if length(word) > 0])
                println(KongYiji.IcwbScoreTable(dict, z, y, source))
                println("==================================")
                
        end
end
