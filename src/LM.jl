using Statistics: mean
using Base.Iterators: cycle
using IterTools: ncycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, dropout, value, nll, 
        adam, minibatch, progress!, progress, converge, cpucopy, logsoftmax, softmax

struct Embed; w; end

Embed(embed::Int, vocab::Int)=Embed(param(embed,vocab))

(e::Embed)(x) = e.w[:,x]

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * mat(x,dims=1) .+ l.b  # (H,B,T)->(H,B*T)->(V,B*T)

struct LM
        embed::Embed
        rnn::RNN
        linear::Linear
end

function (m::LM)(x)
        x = m.embed(x)
        x = m.rnn(x)
        x = m.linear(x)
        x
end

(m::LM)(x,y) = nll(m(x),y)
(m::LM)(d::Data) = mean(m(x,y) for (x,y) in d)

# The h=0,c=0 options to RNN enable a persistent state between iterations
LM(embedsz::Int, vocabsz::Int, hiddensz::Int, outputsz::Int; o...) = 
    LM(Embed(embedsz, vocabsz), RNN(embedsz,hiddensz;h=0,c=0,o...), Linear(hiddensz, outputsz))

LM(name::String) = Knet.load(KongYiji.dir("lm.jld2"), name)

function prepare(xs::Vector{<:Integer}, batchsz, seqlen)
        nbatch = length(xs) ÷ batchsz
        x = reshape(xs[1:nbatch * batchsz], nbatch, batchsz)' # reshape full data to (B,N) with contiguous rows
        minibatch(x[:, 1:nbatch - 1], x[:, 2:nbatch], seqlen) # split into (B,T) blocks 
end

function train!(m::LM, seq::Vector{<:Integer}, epochs, batchsz, seqlen)
        data = prepare(seq, batchsz, seqlen)
        losses = [begin
                        reset!(m, (0,0))
                        loss
                  end
                  for loss in every(progress(adam(m, ncycle(data, epochs))), length(data))
        ]
        Knet.gc()
        return losses
end

every(itr, n) = (x for (i,x) in enumerate(itr) if i % n == 0);

hiddens(m::LM) = (m.rnn.h, m.rnn.c)

function reset!(m::LM, hc)
        m.rnn.h = hc[1]
        m.rnn.c = hc[2]
end

function LM(corpus; o...)
        seqs = posidsents(corpus)
        np = length(postags(corpus))
        embedsz::Int = np
        vocabsz::Int = np
        hiddensz::Int = np #todo
        outputsz::Int = np
        epochs::Int = 100
        batchsz::Int = 256
        seqlen::Int = 100

        nn = LM(embedsz, vocabsz, hiddensz, outputsz; o...)
        seq = collect(Iterators.flatten(seqs))
        train!(nn, seq, epochs, batchsz, seqlen)
        #nn = nn |> cpucopy
        reset!(nn, (0, 0))
        return nn
end

function generate!(lm::LM, start::Int; words=nothing, len=100)
        
        r = String[]
        getword(i) = isnothing(words) ? string(i) : words[i]
        x = start
        for _ in 1:len
                push!(r, getword(x))
                py = softmax(lm(x))
                rd, acc = rand(), 0.
                for (i, p) in enumerate(py)
                        acc += p
                        if rd <= acc; x = i; break; end
                end
        end
        r
end





























