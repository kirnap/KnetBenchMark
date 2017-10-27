using Knet, ArgParse
include("params_lstm.jl")


if VERSION >= v"0.6.0"
    @eval relu_dot(x) = relu.(x)
    @eval sigm_dot(x) = sigm.(x)
    @eval tanh_dot(x) = tanh.(x)
else
    @eval relu_dot(x) = relu(x)
    @eval tanh_dot(x) = tanh(x)
    @eval sigm_dot(x) = sigm(x)
end

# Preprocessing files for different data types
# To preprocess the pre-loaded data
# The reason why we preloaded the data is it was saved txt files
# x train matrix needs to be manipulated before-hand
let
    f(x::AbstractString, y::AbstractString) = map(x->convert(Int, x), readdlm("$y/$x"))
    nz(t::Array{Int64, 1}) = sum(t .== 0)
    function correct_data(data::Array)
        x_correct = fill!(similar(data), 0)
        for i in 1:size(data)[1]
            nzeros = nz(data[i,:])
            x_correct[i, nzeros+1:end] = data[i, 1:end-nzeros]
        end
        return x_correct
    end
    global imdb_for_library
    function imdb_for_library(;datadir="RNN")
        x_train = correct_data(f("x_train.txt", datadir))
        y_train = f("y_train.txt", datadir)
        info("Train data loaded")
        info("Test data loaded")
        x_test = correct_data(f("x_test.txt", datadir))
        y_test = f("y_test.txt", datadir)
        return (x_train, x_test, y_train, y_test)
    end
end


# Minibatch 1 assumes correct minibatching mechanism for sequence labeling
# When testing whether it is correctly batched or do not apply the f function in line 30
let f(x::Int) = (x==0 ? 2 : x); global minibatch1
    function minibatch1(corpus, labels, batchsize; usegpu=true, major=:col)
        data, masks, ygolds = Any[], Any[], Any[]
        indix = randperm(length(labels))
        for i in 1:batchsize:size(corpus)[1]
            j = min(i+batchsize-1, size(corpus)[1])
            batch = corpus[indix[i:j], :]
            push!(ygolds, labels[indix[i:j]])
            sequences = Any[]
            _mask = Any[]
            for r in 1:size(batch)[2]
                mask = ones(Float32, 1, (j-i+1))
                mask[find(t->t==0, batch[:, r])] = 0.0
                s1 = map(f, batch[:, r])
                push!(sequences, s1)
                m1 = (usegpu ? convert(KnetArray, mask) : mask )
                m1 = (major == :col ? m1 : transpose(m1))
                push!(_mask, m1)
            end
            push!(data, sequences)
            push!(masks, _mask)
        end
        return data, masks, ygolds
    end
end


function initmodel(embed_size, hidden_size, vocab_size, outsize; init=randn, ftype=Float32, major=:col, usegpu=true)
    model = Any[]
    f = (usegpu ? KnetArray{ftype} : Array{ftype})
    c = (init == randn ? 0.1 : 1)
    trasp(x) = (major == :col ? x : transpose(x))
    compound_init(x...) = trasp(f(c*init(x...)))
    push!(model, compound_init(embed_size, vocab_size)) # embedding
    push!(model, compound_init(outsize, hidden_size)) # soft_w
    push!(model, trasp(f(zeros(outsize, 1)))) # soft_b
    push!(model, compound_init(2hidden_size, hidden_size+embed_size)) # gru_w1
    push!(model, compound_init(hidden_size, hidden_size+embed_size)) # gru_w2
    return model
end

#implement accessors
embed(model) = model[1]
soft_w(model) = model[2]
soft_b(model) = model[3]
GRU_w1(model) = model[4]
GRU_w2(model) = model[5]


# column based gru implementation
function gru(weight1, weight2, hidden, input)
    gates = sigm_dot(weight1*vcat(input, hidden))
    H = size(hidden, 1)
    z = gates[1:H, :]
    r = gates[1+H:2H, :]
    change = tanh_dot(weight2 * vcat(r .* hidden, input))
    hidden = (1 .- z) .* hidden + z .* change

    #masking operation, in case mask needed
    # if mask !=nothing
    #     hidden = hidden .* mask
    # end
    return hidden
end


function logprob(outputs, ypred)
    nrows, _ = size(ypred)
    index = similar(outputs)
    @inbounds for i in 1:length(outputs)
        index[i] = (outputs[i]+1) + (i-1)*nrows
    end
    o1 = logp(ypred, 1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end 


function gruloss(model, state, sequence, ygold; result=nothing)
    embedding, gru_w1, gru_w2  = embed(model), GRU_w1(model), GRU_w2(model)
    soft_W, soft_B = soft_w(model), soft_b(model)
    h = zeros(similar(state[1], size(state[1])[1], length(ygold)))
    for idx in sequence
        input = embedding[:, idx]
        h = gru(gru_w1, gru_w2, h, input)
    end
    ypred = soft_W * h .+ soft_B
    total = logprob(ygold, ypred)
    count = length(ygold)
    val =  -total / count
    
    if result != nothing
        result[1] += AutoGrad.getval(total)
        result[2] += count
    end
    return val
end


grugrad = grad(gruloss)

# optimization parameter creator for parameters
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)


function accuracy(model, xtest, ytest; usegpu=true)
    embedding = embed(model); gru_w1 = GRU_w1(model); gru_w2  = GRU_w2(model)
    soft_W, soft_B = soft_w(model), soft_b(model)
    
    atype = typeof(embedding); 
    ntot = 0; ncorrect = 0; batchsize= length(ytest);

    state = Any[  ]

    ntot=0; ncorrect = 0;
    data, _, ygolds = minibatch1(xtest, ytest, batchsize; usegpu=usegpu, major=:col)
    hidden_size = size(soft_W)[2]
    x = atype(zeros(Float32, hidden_size, batchsize))
    push!(state, x)
    for (sequence, ygold) in zip(data, ygolds)
        h = zeros(similar(state[1], size(state[1])[1], length(ygold)))
        for idx in sequence
            input = embedding[:, idx]
            h = gru(gru_w1, gru_w2, h, input)
        end
        ypred = soft_W * h .+ soft_B
        nrows, _ = size(ypred)
        index = similar(ygold)
        @inbounds for i in 1:length(ygold)
            index[i] = (ygold[i]+1) + (i-1)*nrows
        end
        ntot += length(index)
        ncorrect += (sum(reshape(findmax(Array((logp(ypred, 1))), 1)[2], length(ygold)) .== index))
    end
    return ncorrect/ntot
end


function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "Knet GRU based sentiment analysis model"
    s.exc_handler = ArgParse.debug_handler
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=3; help="Number of training epochs")
        ("--optimization"; default="Adam"; help="Optimization algorithm")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed")
        ("--hidden"; arg_type=Int; default=NUMHIDDEN; help="Number of GRU hidden units")
        ("--embed"; arg_type=Int; default=EMBEDSIZE; help="Number of embedding units")
        ("--batchsize"; arg_type=Int; default=BATCHSIZE; help="Batchsize of model")
        ("--lr"; arg_type=Float64; default=LR; help="Learning rate")
        ("--betas"; nargs='+'; default=[BETA_1, BETA_2]; help="Beta parameters of ADAM")
        ("--eps"; arg_type=Float64; default=EPS; help="Epsilon parameter of ADAM")
        ("--maxfeatures"; arg_type=Int; default=MAXFEATURES; help="Padded sequence length")
        ("--usegpu"; action=:store_true; help="Employing gpu flag")
        ("--major"; arg_type=Int; default=1; help= "1 for col 2 for row major")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println(s.description)
    inoptim = eval(parse(o[:optimization]))
    println("opts=", [(k,v) for (k,v) in o]...)
    if o[:seed] > 0
        srand(o[:seed])
    end
    atype = (o[:usegpu] ? KnetArray : Array)

    info("reading the data...")
    x_train, x_test, y_train, y_test = imdb_for_library();
    odata, _, ygolds = minibatch1(x_train, y_train, o[:batchsize]; usegpu=o[:usegpu], major=:col)

    info("Initializing the model")
    model = initmodel(o[:embed], o[:hidden], o[:maxfeatures], outlabel; usegpu=o[:usegpu], major=:col)
    x = atype(zeros(Float32, o[:hidden], o[:batchsize]))
    state = Any[ atype(zeros(Float32, o[:hidden], o[:batchsize])) ]

    opts = oparams(model, inoptim;lr=o[:lr], beta1=o[:betas][1], beta2=o[:betas][2], eps=o[:eps])
    info("Calculating the accuracies before train start")
    testacc = accuracy(model, x_test, y_test)
    trainacc = accuracy(model, x_train, y_train)
    println("Before training...Accuracies: train: $trainacc test: $testacc")

    info("Training started")
    for epoch=1:o[:epochs]
        result = zeros(2)
        @time for (sequence, ygold) in zip(odata, ygolds)
            grads = grugrad(model, state, sequence, ygold; result=result)
            update!(model, grads, opts)
        end
        lval = -result[1]/result[2]
        testacc = accuracy(model, x_test, y_test)
        trainacc = accuracy(model, x_train, y_train)
        println("Epoch: $epoch Loss: $lval Train acc: $trainacc Test acc: $testacc")
    end
end
!isinteractive() && main(ARGS)
