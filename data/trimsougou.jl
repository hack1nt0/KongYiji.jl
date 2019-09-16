
function f()
         for file in readdir()
                         file2 = file * "2"
                         open(file2, "w") do io
                                 for line in eachline(file)
                                         println(io, split(line)[end])
                                 end
                         end
                         cp(file2, file; force=true)
                         rm(file2)
         end
end

cd(f, joinpath(pwd(), "usrdict", "sogou"))
