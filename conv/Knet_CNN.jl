# CNN model implementation, by Omer Kirnap
# The model is adapted from:
# https://github.com/ilkarman/DeepLearningFrameworks
using PyCall, JLD, Knet
const LR = 0.01
const MOMENTUM = 0.9

if VERSION >= v"0.6.0"
    @eval relu_dot(x) = relu.(x)
else
    @eval relu_dot(x) = relu(x)
end


# optimization parameter creator for parameters
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict([ k=>oparams(v,otype;o...) for (k,v) in a ])
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)


function prepare_data(;ddir="../data/CNN/CNN_data.jld")
    info("Preparing data")
    x_train, x_test, y_train, y_test = nothing, nothing, nothing, nothing
    try
        x_train, x_test, y_train, y_test =
            JLD.load(ddir, "x_train", "x_test", "y_train", "y_test")
    catch e
        println("Couldn't find preloaded cifar data")

        # That line helps to import your own module
        unshift!(PyVector(pyimport("sys")["path"]), "")
        @pyimport get_data
        x_train, x_test, y_train, y_test = get_data.cifar_for_library(channel_first=true)
    finally
        ispath("cifar-10-batches-py/") && rm("cifar-10-batches-py/"; force=true, recursive=true);
    end
    # This line is needed to make conventional KnetConv operations(sp1, sp2, channel, bsize)
    x_train2 = permutedims(x_train, [3,4,2,1])
    x_test2  = permutedims(x_test, [3,4,2,1])
    return x_train2, x_test2, y_train, y_test
end


function initmodel(;init=xavier, usegpu=true, ftype=Float32)
    f = (usegpu ? KnetArray{ftype} : Array{ftype})
    model = Any[]
    push!(model, f(init(3, 3, 3, 50))) # conv1
    push!(model, f(zeros(1,1, 50, 1))) # bias1
    
    push!(model, f(init(3, 3, 50, 50))) # conv2
    push!(model, f(zeros(1,1, 50,1))) # bias2
    
    push!(model, f(init(3, 3, 50, 100))) # conv3
    push!(model, f(zeros(1,1,100,1))) # bias3

    push!(model, f(init(3,3, 100, 100))) # conv4
    push!(model, f(zeros(1,1,100,1))) # bias4

    push!(model, f(init(512, 6400))) # fc1
    push!(model, f(zeros(512, 1))) # bias

    push!(model, f(init(10, 512))) #soft_w
    push!(model, f(zeros(10, 1)))  # soft_b
    return model
end


function logprob(outputs, ypred)
    nrows, ncols = size(ypred)
    index = similar(outputs)
    @inbounds for i in 1:length(outputs)
        index[i] = (outputs[i] + 1) + (i-1)*nrows
    end
    o1 = logp(ypred, 1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end



function predict(model, x, ygold; odrop=(0.25, 0.25, 0.5))
    _conv1 = conv4(model[1], x;padding=1)
    conv1_2 = _conv1 .+ model[2]
    rel1 = relu_dot(conv1_2)

    
    _conv2 = conv4(model[3], rel1; padding=1)
    conv2_2 = _conv2 .+ model[4]

    pool1 = pool(conv2_2)

    relu2 = relu_dot(pool1)
    drop1 = dropout(relu2, odrop[1])

    _conv3 = conv4(model[5], drop1; padding=1)
    conv3_3 = _conv3 .+ model[6]

    relu3 = relu_dot(conv3_3)
    _conv4 = conv4(model[7], relu3;padding=1)
    conv4_4 = _conv4 .+ model[8]

    pool2 = pool(conv4_4)

    relu4 = relu_dot(pool2)
    drop2 = dropout(relu4, odrop[2])

    flatten = mat(drop2)
    fc1 = relu_dot(model[9] * flatten .+ model[10])
    drop3 = dropout(fc1, odrop[3])
    ypred = model[11] * drop3 .+ model[12]
    return ypred
end


function cnnloss(model, x, ygold; odrop=(0.25, 0.25, 0.5), result=nothing)
    ypred = predict(model, x, ygold; odrop=(0.25, 0.25, 0.5))
    total = logprob(ygold, ypred)
    count = length(ygold)
    if result !=nothing
        result[1] += AutoGrad.getval(total)
        result[2] += count
    end
    return -total / count
end

cnngrad = grad(cnnloss)

function accuracy(model, X, y)
    data = minibatch(X, y;batchsize=500)
    ntot = ncorrect = 0
    for (x, ygold) in data
        ypred = predict(model, x, ygold; odrop=(0, 0, 0))
        nrows, ncols = size(ypred)
        index = similar(ygold)
        @inbounds for i in 1:length(ygold)
            index[i] = (ygold[i] + 1) + (i-1)*nrows
        end
        ntot += length(index)
        ncorrect += (sum(reshape(findmax(Array((logp(ypred, 1))), 1)[2], length(ygold)) .== index))
    end
    return ncorrect /ntot
    
end


function minibatch(X, y; batchsize=64, usegpu=true)
    indix = randperm(length(y))
    data = Any[]
    for i in 1:batchsize:size(X)[4]
        j = min(i+batchsize-1, size(X)[4])
        batch = X[:, :, :, indix[i:j]]
        ygold = y[indix[i:j]]
        trial = (usegpu ? KnetArray{Float32}(batch) : batch)
        push!(data,(trial, ygold))
    end
    return data
end


function train!(model, data, opts)
    result = zeros(2)
    @time for (x, y) in data
        grads = cnngrad(model, x, y;result=result)
        update!(model, grads, opts)
    end
    return - result[1] / result[2]
end


function main()
    x_train, x_test, y_train, y_test = prepare_data()
    model = initmodel();
    opts = oparams(model, Momentum;lr=LR, gclip=0, gamma=MOMENTUM)
    data = minibatch(x_train, y_train)
    for epoch =1:10
        shuffle!(data)
        train!(model, data, opts)
        acc1 = accuracy(model, x_train, y_train)
        acc2 = accuracy(model, x_test, y_test)
        println(":epoch $epoch :acc1 $acc1 :acc2 $acc2")
    end

end
!isinteractive() && main()
