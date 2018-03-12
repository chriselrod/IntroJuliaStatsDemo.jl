module IntroJuliaStatsDemo

# Actually using.
using Compat, Revise
# These are used in the examples, by having the module load (but not export) them, they will load faster later.
using DataFrames, Distributions, GLM, RCall, ForwardDiff 

export  meaning_of_life

function __init__()
    Revise.track( @__FILE__ )
    ENV["EDITOR"] = "code"
end

# package code goes here
meaning_of_life() = 43


end # module

## Things outside of the module won't get run when using the package.

#You can create vectors of random normals.
x = randn(40);
x'
# Adding the semicolon to the end hides the output.
# Vectors are column vectors by default, which take up a good chunk of screen.
# So often it's nicer to tranpose them.

#Or arrays of random uniforms.
u = rand(3,2,2)

#Standard syntax for randomly sampling is to call rand on the thing you want to sample from.
rand(1:4, 10)

using Distributions
gamma_3_3 = Gamma(3, 3)
rand(gamma_3_3, 2, 4)

#Julia Arrays are like the arrays/vectors/matrices of R. They will also automatically promote
[3f0 4.0 2 π]

#The individual elements were
3f0 |> typeof
4.0 |> typeof
2 |> typeof
π |> typeof
#Note that pi is an irrational number! It automatically converts to whatever you want it to be.
2π
2π |> typeof
big(2)π
big(2)π |> typeof
# |> is the pipe operator in Julia.
# However, it isn't as powerful or fancy as magrittr's "%>%".
# You can look at Lazy.jl if you want more piping options.

#However, if Julia can't promote them all to the same type, it'll give up and create a vector that can hold anything.
[ 42im 9//2 "fish" π ]


#This isn't a good idea. Don't that.

#You can also create undefined arrays. Be cautious, they may contain NaNs, which can infect code if you're not careful!
Array{Float64}(uninitialized, 4, 12)

#Or fill
fill(1.2, 4, 7)
fill(gamma_3_3, 2, 4)

#Or if you just want zeros
zeros(4,7)
zeros(Int, 4, 7)



#You can also use comprehensions. Remember x earlier?
x'
#Lets use it to create a design matrix!
X = [xᵢ^j for xᵢ ∈ x, j ∈ 0:4]

#Lets create some fake data! First, lets get a true beta:
β_true = [2, 3, -5, -2, 1.2]

#And then add a little noise
y = X * β_true .+ 0.1randn(40)

β_hat = X \ y

# "\" is the left devision operator.
# It just means use the thing on the left to divide the thing on the right
2 \ 4
# If we tranpose both X and y, we can get dimensions to line up for a right division
y' / X'


# This is of course equivalent to what we're all used to:
inv(X' * X) * X' * y

# Realstically, we'd always prefer to just use LM and immediately get convenient summary statistics.
using DataFrames, GLM
df = DataFrame(x = x, x² = X[:,3], x³ = X[:,4], x⁴ = X[:,5], y = y)
OLS = glm(@formula(y ~ x + x² + x³ + x⁴), df, Normal(), IdentityLink())


# There are three ways to define functions.
# For an anonymous function, use [arguments] -> [function body]
# For example, if we want a random positive definite matrix
Σ = randn(20,5) |> x -> x' * x

# If we have more than one argument, we need to wrap them in a parenthesis.
anon = (a, b) -> sin(a) - exp(b)
anon(2, 3)

# We can overwrite anon with anything.
anon = 4

# If we create a proper named function, we cannot replace it with something other than a function.
f(a) = 2a
f(4)
f = 3
f(a) = 3a
f(4)
f(a::Int) = "Hello world!"
f(a, b) = 3a + 2b
f(4)
f(4.0)
f(4, -5)

# Julia uses multiple dispatch.
# Julia compiles a different specialized version of that function for each combination of types you give it.
# It then calls the type matching that combination.
# This also lets you define a different body for the types if you want to.
# That is, Julia compiles 4 separate functions for "f" when we do this:
f(4, -5)
f(4.0, -5)
f(4, -5.0)
f(4.0, -5.0)
# And chooses the correct version to use.
# If we want the compiled function to act differently, we can then give Julia a different function body to use for that type
# Just like we did for "Hellow world!" above
# For example:
f(X, β_hat) #Error, dimensions don't line up!
f(a::Matrix, b::Vector) = b' * a' * a * b
f(X, β_hat)
f(4, -5) #Still works!
# Similar to match.call() in R:
pair = (4,-5)
f(pair...)

# The long form just makes it easier to write multi-line functions.
function f(x::Irrational)
    println("No, you're irrational!")
    BigFloat(x)
end
f(π)
f(e)
# In Julia, you end a scope or expression via "end".
f(√2) #Does not have the irrational type! It's just taking the square root of 2.


# R has "single dispatch", which is useful for things like plot, print, and summary methods.

# A simple example why this is useful in Julia is so that we can use "*" for matrix multiplication.
# There are a lot of different methods for "*"!
methods(*) |> length
#Run just methods(*) if you want to see the full list!

# This also makes code faster!
# Compare
S_big = randn(10_000, 5_000) |> x -> x' * x;
X_big = randn(5_000, 5_000);
@time S_big * X_big;
using Compat.LinearAlgebra #This line is currently unnecessary
@time eigfact(S_big);
# However, we know S is symmetric! What happens if we tell Julia that?
S_big2 = Symmetric(S);
@time S_big2 * X_big;
@time eigfact(S_big2);

#The using Compat.LinearAlgebra line is unnecassary in Julia 0.6
#However, starting in Julia 0.7 you'll need "using LinearAlgebra"
#Saying "using Compat.LinearAlgebra" works on both versions.

# Give Julia more information, and it will automatically pick the best method!

# Multiple dispatch is magical, because
# a) It makes code easier to read. When you want to multiply, you use "*".
# When you want an eigenfactorization, you use "eigfact".
# No "eigfact" here, "eigfact_symmetric" there.
# No map_int here and map_dbl there.

# However, what if we actually want elementise multiplication from "*"?
# In Julia, adding a "." automatically broadcosts.
# A broadcast is like an apply or map statement, except it will line up dimensions.

# Recall
size(X)
size(y)
y' * X # Regular matrix-vector product.
X .* y # Broadcast
# Both X and y have 40 rows, so it lines these up
# and then replicates y across each of X's 5 columns.
X .* β_hat'
# β_hat' is 1 x 5, so it lines up with X's 5 columns, and is replicated down the 40 rows.
X .+ 2
# Here, 2 gets replcicated along both colums and rows.

X .* y' # Does not line up -> error
X .* β_hat # Does not line up -> error

# This works for any function.
f.(X)
f.(X, y)

# If we use multiple ones, they will all fuse.
f.(X, y) .* β_hat' .+ 2

# Too many dots can be annoying to write. So Julia offers the macro "@. "
@. f(X, y) * β_hat + 2

# Whenever you see something with an "@" in Julia, it is a macro.
# Macros edit your code. Think of them as non-standard evaluation in R.
# Because they can do extremely weird things, they always have an "@" to let you know something funky is going on.
# A basic example of non-standard evaluation in R is how a lot of functions know the name of the variables you give them.
# Consider -- straight out of Advanced R:
"
x <- seq(0, 2 * pi, length = 100)
sinx <- sin(x)
plot(x, sinx, type = "l")
"
# Similarly:
@show anon;
# If you ever want to know what it is a macro does to your code, there's a macro for that!
@macroexpand @show anon
@macroexpand @. f(X, y) * β_hat + 2

# We can see that "@." does us the favor of adding "." to every function call for us.

# We can get creative!
bunch_o_gammas = @. Gamma(exp(f(X)), exp(-2y))

# What is the meaning of this? Good question!
mean.(bunch_o_gammas)
# Maybe we want to do parametric bootstrap?
samples = rand.(bunch_o_gammas, 4000)

# Method of Moment's estimator for Gamma
function gammas_MoM(x)
    x̄ = mean(x)
    s² = var(x)
    x̄^2 / s², s² / x̄
end
mom_estimators = gammas_MoM.(samples)
function relative_error( estimate, truth )
    @. ( estimate - truth ) / truth
end
#Relative errors
@. relative_error(mom_estimators, params(bunch_o_gammas))

#We can easily condense the steps
N = 4000
@. relative_error(gammas_MoM(rand(bunch_o_gammas, N)), params(bunch_o_gammas))



### Some more array manipulations and fun with broadcasting!
# Remember Σ from earlier?
Σ
# Let's take N samples from a multivariate normal with that as the covariance matrix, and mean:
μ = 2.* randn(5)

W = chol(Σ) * randn(5, N) .+ μ

mean(W, 2)
cov(W')
# We can also recall the definition from multivariate
S = (W .- mean(W,2)) |> x -> x * x' ./ (size(x,2)-1)
#and see that it is a pretty good estimator
Σ
# The matrix and broadcast syntax makes it easy to weave together array manipulations
# and apply definitions, like we see in Johnson and Wichern for the sample Covariance matrix.


# More cool features
using ForwardDiff

ForwardDiff.derivative(f, 2.3)
ForwardDiff.derivative(sin, 2.3)
cos(2.3)
