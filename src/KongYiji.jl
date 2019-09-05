module KongYiji

import Base: size, split, display, read, write, first, stat, push!, length, iterate, ==, getindex, summary, collect, values, isless, get, eachmatch, in

using Pkg
using JLD2, FileIO
using Random, DataStructures, ProgressMeter, DelimitedFiles, DataFrames

export postable, h2vtable, v2htable, h2htable, hprtable, kwictable,
        ChTreebank, CtbDocument, CtbSentence, train!, HMM2


const Ti = UInt32
const Tv = Float32

include("CtbTree.jl")
include("ChTreebank.jl")
include("utils.jl")
include("UselessTable.jl")
include("compress.jl")
include("AhoCorasickAutomaton.jl")
include("HMM.jl")
include("HmmScoreTable.jl")
include("HmmDebug.jl")
include("IcwbScoreTable.jl")
include("PosLM.jl")
include("ooHMM.jl")

end # module
