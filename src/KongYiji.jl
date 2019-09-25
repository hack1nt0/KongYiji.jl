module KongYiji

import Base: size, split, display, read, write, first, stat, push!, length, iterate, ==, getindex, summary, collect, values, isless, get, eachmatch, in

using Pkg
using JLD2, FileIO
using Random, DataStructures, ProgressMeter, DelimitedFiles, DataFrames

export postable, h2vtable, v2htable, h2htable, hprtable, kwictable,
        ChTreebank, ZhuXian, posidsents, wordsents, rawsents,
        UselessTable, HMM, NGramHMM

#=
export generate!, train!, LM, LMHMM
=#

const Ti = UInt32
const Tv = Float32
const oo = Tv(Inf)
const EPS = Tv(1e-9)

include("utils.jl")

include("CtbTree.jl")
include("ChTreebank.jl")

include("ZhuXian.jl")

include("UselessTable.jl")
include("compress.jl")
include("AhoCorasickAutomaton.jl")

include("HmmUtils.jl")
include("HmmScoreTable.jl")
include("IcwbScoreTable.jl")
include("HMM.jl")
include("NGramHMM.jl")


#include("LM.jl")
#include("Heap.jl")
#include("LMHMM.jl")


end # module
