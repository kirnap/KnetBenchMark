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
    function imdb_for_library(;datadir="data/RNN")
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
