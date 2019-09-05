using Flux
using Flux: gate, batchseq, throttle, glorot_uniform, @treelike
using Base.Iterators: partition

mutable struct HMM_LSTM_Cell
        Wh
        b
        h
        c
end

import Flux.hidden
hidden(m::HMM_LSTM_Cell) = (m.h, m.c)

function HMM_LSTM_Cell(hmm::HMM; init = glorot_uniform)
        npos, nword = length(hmm.tags), length(hmm.words)
        out = npos
        cell = HMM_LSTM_Cell(param(init(out*4, out)), param(init(out*4)),
                             param(zeros(out)), param(zeros(out)))
        cell.b.data[gate(out, 2)] .= 1 #todo ?
        cell.h.data .= hmm.hpr
        return cell
end


@treelike HMM_LSTM_Cell #todo ?

function (m::HMM_LSTM_Cell)((h, c), x)
        b, o = m.b, size(h, 1)
        #@show size.([m.Wh, h, b])
        #@show x
        g = m.Wh * h .+ b
        input = σ.(gate(g, o, 1))
        forget = σ.(gate(g, o, 2))
        cell = tanh.(gate(g, o, 3))
        output = σ.(gate(g, o, 4))
        c = forget .* c .+ input .* cell
        h′ = output .* tanh.(c)
        return (h′, c), h′
end


HMM_LSTM(a) = Flux.Recur(HMM_LSTM_Cell(a))


function trhmmlstm()
        ctb = ChTreebank()
        xs = mapreduce(doc -> map(tokens, doc), append!, ctb)
        hmm = HMM(ctb)
        normalize!(hmm)
        ys = map(sent -> map(word -> Int(hmm.dict[word]), sent), xs)
        #xs = map(sent -> map(word -> 0, sent), ys)
        #xs = [[0,0], [0,0], [0,0]]
        #ys = [[1,2], [3,4], [5,6]]
        nsent = length(xs)
        ntrain = Int(round(nsent * 0.8))

        xs_tr, xs_te = xs[1:ntrain], xs[ntrain+1:end]
        ys_tr, ys_te = ys[1:ntrain], ys[ntrain+1:end]
        batch_size = 10
        ys_tr = collect(partition(ys_tr, batch_size))
        xs_tr = collect(partition(xs_tr, batch_size))

        m = Chain(
                HMM_LSTM(hmm),
                softmax
        )

        #m = gpu(m)

        function loss(xs, ys)
                r = 0.
                for i in 1:length(xs)
                        probs, y = m.(xs[i]), ys[i]
                        for (j, prob) in enumerate(probs)
                                r1 = 0.
                                for k in 1:length(prob)
                                        r1 += prob[k] * get(hmm.h2v[k], y[j], hmm.INF[k])
                                end
                                r += log(r1)
                        end
                end
                Flux.truncate!(m) #todo ?
                return -r
        end

        opt = ADAM(0.01)
        evalcb = () -> begin
                @show loss(xs_te[1:1], ys_te[1:1])
        end

        Flux.train!(loss, params(m), zip(xs_tr, ys_tr), opt,
                    cb = throttle(evalcb, 2))

end