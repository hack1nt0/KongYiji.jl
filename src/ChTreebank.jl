
struct CtbSentence
        tree::String
        poswords::Vector{Tuple{Ti, Ti}}
end

struct CtbDocument
        type::String
        sents::Vector{CtbSentence}
end

struct ChTreebank
        postags::Vector{String}
        inntags::Vector{String}
        words::Vector{String}
        docs::Vector{CtbDocument}
end

postags(ctb::ChTreebank) = ctb.postags
inntags(ctb::ChTreebank) = ctb.inntags
words(ctb::ChTreebank) = ctb.words

import Base.show
function show(io::IO, ctb::ChTreebank)
        nd = length(ctb)
        ns = mapreduce(length, +, ctb)
        nw = mapreduce(length, +, Iterators.flatten(ctb))
        print(io, "Chinese Treebank of $(nd) docs, $(ns) sents, $(length(ctb.words))/$(nw) words")
end

function ChTreebank(home::String; nf=0)
        home_data = joinpath(home, "data", "bracketed")
        if (nf <= 0) nf = length(readdir(home_data)) end
        docs = Vector{CtbDocument}(undef, nf)
        postagMap = Dict{String, Ti}()
        inntagMap = Dict{String, Ti}()
        wordMap = Dict{String, Ti}()
        @showprogress 1 "Parsing ChTreebank..." for (i, file_name) in enumerate(readdir(home_data))
                if i > nf break end
                type = ""
                id = parse(Int, file_name[6:9])
                if 0001<=id<=0325 || 0400<=id<=0454 || 0500<=id<=0540 || 0600<=id<=0885 || 0900<=id<=0931 || 4000<=id<=4050 type = "Newwire"
                elseif 0590<=id<=0596 || 1001<=id<=1151 type = "Magazine articles"
                elseif 2000<=id<=3145 || 4051<=id<=4111 type = "Broadcast news"
                elseif 4112<=id<=4197 type = "Broadcast conversations"
                elseif 4198<=id<=4411 type = "Weblogs"
                elseif 5000<=id<=5558 type = "Discussion forums"
                else type = "N/A"
                end
                treestrs = parsectbfile(joinpath(home_data, file_name))
                sents = CtbSentence[]
                for treestr in treestrs
                        tree = CtbTree(treestr)
                        pws = poswords(tree)
                        pws = map(p -> (getid!(postagMap, p[1]), getid!(wordMap, p[2])), pws)
                        push!(sents, CtbSentence(String(treestr), pws))
                        
                        for inntag in inntags(tree) getid!(inntagMap, inntag) end
                end
                docs[i] = CtbDocument(type, sents)
        end
        postags, _inntags, words = vec.((postagMap, inntagMap, wordMap))
        r = ChTreebank(postags, _inntags, words, docs)
        return r
end

function ChTreebank()
        file = KongYiji.dir("ctb.jld2")
        zfile = KongYiji.dir("ctb.jld2.7z")
        if isfile(file)
                r = load(file)["ctb"]
        elseif isfile(zfile)
                r = load(KongYiji.decompress(zfile))["ctb"]
        else
                error("CTB not downloaded.")
        end
        return r
end
        

Base.length(sent::CtbSentence) = length(sent.poswords)
Base.length(doc::CtbDocument) = length(doc.sents)
Base.length(ctb::ChTreebank) = length(ctb.docs)
Base.getindex(sent::CtbSentence, inds...) = getindex(sent.poswords, inds...)
Base.getindex(doc::CtbDocument, inds...) = getindex(doc.sents, inds...)
Base.getindex(ctb::ChTreebank, inds...) = getindex(ctb.docs, inds...)
Base.iterate(sent::CtbSentence, state=1) = state > length(sent) ? nothing : (sent.poswords[state], state + 1)
Base.iterate(doc::CtbDocument, state=1) = state > length(doc) ? nothing : (doc.sents[state], state + 1)
Base.iterate(ctb::ChTreebank, state=1) = state > length(ctb) ? nothing : (ctb.docs[state], state + 1)

mutable struct Block
    chrs::Vector{Char}
    nlb::Int
end

function Block()
        chrs = Char[]
        sizehint!(chrs, 100)
        #resize!(chrs, 100)
        return Block(chrs, 0)
end

function push!(b::Block, chrs::Vector{Char})
    for c in chrs
        if c == '('
            b.nlb += 1
        elseif c == ')'
            b.nlb -= 1
        end
        push!(b.chrs, c)
    end
end

function text(b::Block)
        resize!(b.chrs, length(b.chrs));
        return b.chrs
end

function ok(block::Block)
    return block.nlb == 0
end

function parsectbfile(file_path)
        ret = Vector{Char}[]
        b = Block()
        open(file_path, "r") do io
            for line in eachline(io)
                if startswith(line, "(")
                    push!(b, collect(line))
                    if !ok(b)
                        for line in eachline(io)
                            push!(b, collect(line))
                            if ok(b) break end
                        end
                    end
                    push!(ret, text(b))
                    b = Block()
                end
            end
        end
        return ret
end

text(tree::CtbTree) = tree.label
ispostag(tree::CtbTree) = length(tree.adj) == 1 && isleaf(tree.adj[1])

function poswords(sent::CtbTree)
        ret = Tuple{String, String}[]
        visitor(tree::CtbTree) = if ispostag(tree) && text(tree) != "-NONE-" push!(ret, (text(tree), text(tree.adj[1]))) end
        dfstraverse(sent, visitor)
        return ret
end

sents(ctb::ChTreebank) = Iterators.flatten(ctb)

function inntags(sent::CtbTree)
        ret = String[]
        visitor(tree::CtbTree) = if !ispostag(tree) && !isleaf(tree); push!(ret, text(tree)); end
        dfstraverse(sent, visitor)
        return ret
end

function rawsents(ctb::ChTreebank)
        r = [join([ctb.words[p[2]] for p in sent]) for sent in sents(ctb)]
        r
end

function wordsents(ctb::ChTreebank)
        r = [[ctb.words[p[2]] for p in sent] for sent in sents(ctb)]
        r
end

function posidsents(ctb::ChTreebank)
        r = [[p[1] for p in sent] for sent in sents(ctb)]
        r
end

function ==(a::ChTreebank, b::ChTreebank)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(ChTreebank))
end

function ==(a::CtbDocument, b::CtbDocument)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(CtbDocument))
end

function ==(a::CtbSentence, b::CtbSentence)
        return all(fname -> getfield(a, fname) == getfield(b, fname), fieldnames(CtbSentence))
end

function split(ctb::ChTreebank; percents::Vector{Float64}=[0.7, 0.2, 0.1])
        percents ./= sum(percents)
        n = length(ctb)
        caps = map(p -> floor(p * n), percents)
        caps[3] += n - sum(caps)
        idx = randperm(n)
        train = ChTreebank(ctb.tags, ctb.docs[1:caps[1]])
        dev = ChTreebank(ctb.tags, ctb.docs[caps[1]+1:caps[1]+caps[2]])
        test = ChTreebank(ctb.tags, ctb.docs[end-caps[3]+1:end])
        return (train, dev, test)
end

import Base.+
function foldbatch(ctb::ChTreebank, nbatch::Int)
        groups = DefaultDict{String, Vector{CtbDocument}}(()->CtbDocument[])
        for doc in ctb; push!(groups[doc.type], doc); end
        for (_, docs) in groups; shuffle!(docs); end
        nb = min(nbatch, mapreduce(length, max, values(groups)))
        @assert 2 <= nb
        (begin
                tedocs = CtbDocument[]
                trdocs = CtbDocument[]
                for (type, docs) in groups
                        nd = length(docs)
                        npick = div(nd + nb - 1, nb)
                        from = min(nd + 1, (ib - 1) * npick + 1)
                        to = min(nd, from + npick - 1)
                        append!(tedocs, docs[from:to])

                        append!(trdocs, docs[1:from - 1])
                        append!(trdocs, docs[to + 1:nd])
                end
                te = ChTreebank(ctb.postags, ctb.inntags, ctb.words, tedocs)
                tr = ChTreebank(ctb.postags, ctb.inntags, ctb.words, trdocs)
                (tr, te)
        end for ib in 1:nb)
end

function kwictable(;word="", pos="")
        ctb_home = KongYiji.unzip7(joinpath(pathof(KongYiji), "..", "..", "data", "ctb.jld2.7z"))
        ctb = load(ctb_home)["ctb"]
        df = DataFrame(cw=String[], cpos=String[], lw=String[], rw=String[], lpos=String[], rpos=String[])
        for doc in ctb, sent in doc
                nw = length(sent)
                for i in 1:nw
                        cpos, cw = sent[i]
                        if word == cw || pos == cpos
                                #l = i == 1  ? "^" : sent[i - 1][2][end:end]
                                #r = i == nw ? "\$" : sent[i + 1][2][end:end]
                                lpos, lw = i == 1  ? ("^", "") : sent[i - 1]
                                rpos, rw = i == nw ? ("\$", "") : sent[i + 1]
                                push!(df, (cw, cpos, lw, rw, lpos, rpos))
                        end
                end
        end
        return df
end

function mask(ctb::ChTreebank)
        words = mask.(ctb.words) |> sort! |> unique!
        wmp = dict(words)
        nd = length(ctb)
        r = ChTreebank(ctb.postags, ctb.inntags, words, Vector{CtbDocument}(undef, nd))
        for id = 1:nd
                doc1 = ctb.docs[id]
                ns = length(doc1)
                doc2 = CtbDocument(doc1.type, Vector{CtbSentence}(undef, ns))
                for is = 1:ns
                        s1 = doc1[is]
                        s2 = map(p->(p[1], wmp[ctb.words[p[2]] |> mask]), s1)
                        s2 = CtbSentence(s1.tree, s2)
                        doc2.sents[is] = s2
                end
                r.docs[id] = doc2
        end
        r
end
