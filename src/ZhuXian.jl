
const Tsent = Vector{Tuple{Ti, Ti}}

struct ZhuXian
        words::Vector{String}
        postags::Vector{String}
        sents::Vector{Tsent}
end

function ZhuXian(path)
        wmp, pmp = Dict{String, Ti}(), Dict{String, Ti}()
        function go(file)
                sents = Tsent[]
                for line in eachline(file)
                        sent = Tsent()
                        for wordpos in split(line)
                                word, pos = String.(split(wordpos, '_'))
                                iw = getid!(wmp, word)
                                ip = getid!(pmp, pos)
                                push!(sent, (ip, iw))
                        end
                        push!(sents, sent)
                end
                sents
        end
        sents = Tsent[]
        append!(sents, go(joinpath(path, "train.zhuxian.wordpos")))
        append!(sents, go(joinpath(path, "dev.zhuxian.wordpos")))
        append!(sents, go(joinpath(path, "test.zhuxian.wordpos")))

        ZhuXian(vec(wmp), vec(pmp), sents)
end

function ZhuXian()
        file = KongYiji.dir("zhuxian.jld2")
        zfile = KongYiji.dir("zhuxian.jld2.7z")
        if isfile(file)
                r = Knet.load(file, "zx")
        elseif isfile(zfile)
                r = Knet.load(KongYiji.decompress(zfile), "zx")
        else
                error("zhuxian not downloaded.")
        end
        return r
end

import Base.length
length(zx::ZhuXian) = length(zx.sents)

sents(zx::ZhuXian) = zx.sents
words(zx::ZhuXian) = zx.words
postags(zx::ZhuXian) = zx.postags

function rawsents(zx::ZhuXian)
        r = [join([zx.words[p[2]] for p in sent]) for sent in sents(zx)]
        r
end

function wordsents(zx::ZhuXian)
        r = [[zx.words[p[2]] for p in sent] for sent in sents(zx)]
        r
end

function posidsents(zx::ZhuXian)
        r = [[p[1] for p in sent] for sent in sents(zx)]
        r
end

function foldbatch(zx::ZhuXian, nbatch::Int)
        sents = collect(KongYiji.sents(zx)) |> shuffle!
        ns = length(sents)
        nb = min(ns, nbatch)
        @assert 2 <= nb
        (begin
                te = Tsent[]
                tr = Tsent[]
                npick = div(ns + nb - 1, nb)
                from = min(ns + 1, (ib - 1) * npick + 1)
                to = min(ns, from + npick - 1)
                append!(te, sents[from:to])
                append!(tr, sents[1:from - 1])
                append!(tr, sents[to + 1:ns])
                te = ZhuXian(zx.words, zx.postags, te)
                tr = ZhuXian(zx.words, zx.postags, tr)
                (tr, te)
        end for ib in 1:nb)
end

import Base.show
function show(io::IO, zx::ZhuXian)
        ns = length(zx)
        nw = mapreduce(length, +, sents(zx))
        print(io, "Zhu Xian of $(ns) sents, $(length(zx.words))/$(nw) words")
end

function ==(a::ZhuXian, b::ZhuXian)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(ZhuXian))
end

function mask(zx::ZhuXian)
        words = mask.(zx.words) |> sort! |> unique!
        wmp = dict(words)
        ns = length(zx)
        sents = Tsent[]
        for sent in KongYiji.sents(zx)
                sent2 = Tsent()
                for (ip, iw) in sent
                        iw = wmp[mask[zx.words[iw]]]
                        push!(sent2, (ip, iw))
                end
                push!(sents, sent2)
        end
        ZhuXian(words, zx.postags, sents)
end
