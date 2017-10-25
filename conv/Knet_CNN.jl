# CNN model implementation
using PyCall, JLD


function prepare_data()
    info("Preparing data")
    x_train, x_test, y_train, y_test = nothing, nothing, nothing, nothing
    try

        x_train, x_test, y_train, y_test =
            JLD.load("../data/CNN/CNN_data.jld", "x_train", "x_test", "y_train", "y_test")
    catch e
        println("Couldn't find preloaded cifar data")

        # That line helps to import your own module
        unshift!(PyVector(pyimport("sys")["path"]), "")
        @pyimport get_data
        x_train, x_test, y_train, y_test = get_data.cifar_for_library(channel_first=true)
    finally
        ispath("cifar-10-batches-py/") && rm("cifar-10-batches-py/"; force=true, recursive=true);
    end
    @show map(size, [x_train, x_test, y_train, y_test])
    return x_train, x_test, y_train, y_test
end
prepare_data()
