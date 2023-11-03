using Profile
using LinearAlgebra


n = 5
matrix = diagm(ones(5))
X = randn((10, 5))


function f_matrix()
    X * matrix
    :ok
end


function f_rows()
    Ref(matrix) .* eachrow(X)
    :ok
end


f_matrix()


f_rows()



@timev begin
    for i in 1:100
        f_matrix()
    end
end


@timev begin
    for i in 1:100
        f_rows()
    end
end
