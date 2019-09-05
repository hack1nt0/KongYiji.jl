using Statistics: mean
using Base.Iterators: cycle
using IterTools: ncycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, dropout, value, nll, 
        adam, minibatch, progress!, progress, converge, cpucopy, logsoftmax

struct Embed; w; end

Embed(embed::Int, vocab::Int)=Embed(param(embed,vocab))

(e::Embed)(x) = e.w[:,x]

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * mat(x,dims=1) .+ l.b  # (H,B,T)->(H,B*T)->(V,B*T)

struct PosLM
        embed::Embed
        rnn::RNN
        linear::Linear
        epochs::Int
        batchsz::Int
        seqlen::Int
end

function (m::PosLM)(x)
        x = m.embed(x)
        x = m.rnn(x)
        x = m.linear(x)
        x
end

(m::PosLM)(x,y) = nll(m(x),y)
(m::PosLM)(d::Data) = mean(m(x,y) for (x,y) in d)

# The h=0,c=0 options to RNN enable a persistent state between iterations
PosLM(embedsz::Int, vocabsz::Int, hiddensz::Int, outputsz::Int; o...) = 
    PosLM(Embed(embedsz, vocabsz), RNN(embedsz,hiddensz;h=0,c=0,o...), Linear(hiddensz, outputsz))

function prepare(xs::Vector{<:Integer}, batchsz, seqlen)
        nbatch = length(xs) ÷ batchsz
        x = reshape(xs[1:nbatch * batchsz], nbatch, batchsz)' # reshape full data to (B,N) with contiguous rows
        minibatch(x[:, 1:nbatch - 1], x[:, 2:nbatch], seqlen) # split into (B,T) blocks 
end

function train!(m::PosLM, seqs::Vector{<:Integer})
        data = prepare(seqs, m.batchsz, m.seqlen)
        losses = [begin
                        reset!(m, (0,0))
                        loss
                  end
                  for loss in every(progress(adam(m, ncycle(data, m.epochs))), length(data))
        ]
        Knet.gc()
        return losses
end

every(itr, n) = (x for (i,x) in enumerate(itr) if i % n == 0);

hiddens(m::PosLM) = (m.rnn.h, m.rnn.c)

function reset!(m::PosLM, hc)
        m.rnn.h = hc[1]
        m.rnn.c = hc[2]
end

function PosLM(ctb::ChTreebank; retrain=false, file=KongYiji.dir("poslm.jld2"), o...)
        if retrain || !isfile(file)
                seqs = collect(Iterators.flatten(ctb))
                seqs = collect(Iterators.flatten(seqs))
                seqs = map(first, seqs)
                np, nw = length.((ctb.postags, ctb.words))
                m = PosLM(seqs, np, nw, np, np, 10, 256, 60; o...)
                Knet.save(file, "poslm", m)
        else
                m = Knet.load(file, "poslm")
        end
        return m
end

function PosLM(seqs::Vector{Vector{<:Integer}},
               embedsz::Int, 
               vocabsz::Int, 
               hiddensz::Int, 
               outputsz::Int, 
               epochs::Int, 
               batchsz::Int, 
               seqlen::Int,
               o...)
        nn = PosLM(embedsz, vocabsz, hiddensz, outputsz, epochs, batchsz, seqlen; o...)
        train!(nn, seqs)
        nn = nn |> cpucopy
        reset!(nn, (0, 0))
        return nn
end
                


































