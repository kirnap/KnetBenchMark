# Preprocessing files for different data types

# To preprocess the pre-loaded data
# The reason why we preloaded the data is it was saved txt files
function imdb_for_library(;datadir="data/RNN")
    f(x) = map(x->convert(Int, x), readdlm("$datadir/$x"))
    x_train = f("x_train.txt")
    x_test = f("x_test.txt")
    y_train = f("y_train.txt")
    y_test = f("y_test.txt")
    return (x_train, x_test, y_train, y_test)
end
