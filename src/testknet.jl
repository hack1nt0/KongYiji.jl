using Knet
const embsize = 256
const hsize = 256
const bsize = 7
const vocabsize = 40
x = [2, 2, 2, 2, 2, 2, 2, 25, 15, 27, 25, 31, 13, 23, 10,
 17, 10, 21, 10, 10, 3, 7, 24, 13, 3, 17, 19, 15, 10, 3, 30,
 15, 19, 14, 10, 7, 3, 15, 14, 14, 16, 15, 15, 15, 8, 11, 19, 10, 3, 3, 19, 10, 22, 3]

bs = [7, 7, 7, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]


function initstates(hsize,bsize)
    h  = KnetArray(rand(Float32,hsize,bsize))
    c  = KnetArray(rand(Float32,hsize,bsize))
    return h,c
end

function initweights()
    weights = Any[]
    rsettings = Any[]
    emb = KnetArray(rand(Float32,embsize,vocabsize))
    push!(weights,emb)
    r,w = rnninit(embsize,hsize)
    push!(weights,w)
    push!(rsettings,r)
    return weights,rsettings
end


function loss(w,rsettings,x)
    h,c = initstates(hsize,bsize)
    lstm_input = w[1][:,x]
    y,h,c,rs = Knet.rnnforw(rsettings[1],w[2],lstm_input,h,c,batchSizes= bs)
    #=
        # If I call the following line instead of the above, 
        # I don't get any error in training
        # but I need  cell values too.  Is there a way to get final cell values in 
        # this scenario ? 
        y,h,c,rs = rnnforw(rsettings[1],w[2],lstm_input,batchSizes= bs)
    =#
    # placeholder loss value
    return sum(y)
end


lossgradient = grad(loss)

function test(weights,rsettings)
    loss(weights,rsettings,x)
end

function train(weights,rsettings)
    lossgradient(weights,rsettings,x)
end

weights,rsettings  = initweights()

test(weights,rsettings) # works fine
train(weights,rsettings) # causes error 
