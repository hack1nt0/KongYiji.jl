module KongYiji

import Base: size, split, display, read, write, first, stat, push!, length, iterate, ==, getindex, summary, collect, values, isless, get, eachmatch, in

using Pkg
using JLD2, FileIO
using Random, DataStructures, ProgressMeter, DelimitedFiles, DataFrames

export postable, h2vtable, v2htable, h2htable, hprtable, kwictable,
        ChTreebank, train!, HMM, ooHMM, LM, ZhuXian, posidsents, wordsents, rawsents,
        UselessTable, generate!


const Ti = UInt32
const Tv = Float32
const oo = Tv(Inf)

include("utils.jl")

include("CtbTree.jl")
include("ChTreebank.jl")

include("ZhuXian.jl")

include("UselessTable.jl")
include("compress.jl")
include("AhoCorasickAutomaton.jl")
include("HMM.jl")
include("HmmScoreTable.jl")
include("HmmDebug.jl")
include("IcwbScoreTable.jl")
include("LM.jl")
include("ooHMM.jl")

end # module
