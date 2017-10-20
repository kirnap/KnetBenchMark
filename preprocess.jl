# Preprocessing files for different data types

# To preprocess the pre-loaded data
# The reason why we preloaded the data is it was saved txt files
let
    f(x::AbstractString, y::AbstractString) = map(x->convert(Int, x), readdlm("$y/$x"))
    nz(t::Array{Int64, 1}) = sum(t .== 0)
    function correct_data(data::Array)
        x_correct = fill!(similar(data), 0)
        for i in 1:size(x_train)[1]
            nzeros = nz(x_train[i,:])
            x_correct[i, nzeros+1:end] = x_train[i, 1:end-nzeros]
        end
        return x_correct
    end
    global imdb_for_library
    function imdb_for_library(;datadir="data/RNN")
        x_train = correct_data(f("x_train.txt", datadir))
        x_test = correct_data(f("x_test.txt", datadir))
        y_train = f("y_train.txt", datadir)
        y_test = f("y_test.txt", datadir)
        # Here we need to manipulate the x_train matrix
        return (x_train, x_test, y_train, y_test)
    end
end
