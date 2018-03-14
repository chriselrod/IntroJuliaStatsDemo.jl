
## Things outside of the module won't get run when using the package.

#You can create vectors of random normals.
x = randn(40);
x'
# Adding the semicolon to the end hides the output.
# Vectors are column vectors by default, which take up a good chunk of screen.
# So often it's nicer to tranpose them.

#You index into arrays with brackets, like in R
x[3]
x[end]
x[2:5]
x[end-4:end-1]
x[36:end-1]

#You also index into strings that way.
greeting = "Hello world!"
greeting[3:4]
greeting[7:end-1]

#Or arrays of random uniforms.
u = rand(3,2,2)
u[:,:,1]

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

#In "R" a scalar is really a length 1 vector.
#In julia, they are clearly distinct.
scalar4double = 4.0
zerodim4double = fill(4.0)
vector4double = fill(4.0, 1)
matrix4double = fill(4.0, 1, 1)
dim3array4double = fill(4.0, 1, 1, 1)

#They all print differently, but there's also a fundamental difference under the hood.
isbits(scalar4double)
isbits(zerodim4double)
isbits(vector4double)

#Q: Are only numbers bits?
isbits(gamma_3_3)

#"isbits" is low level, referring to how the memory is handled under the hood.
#"isbits" types are raw, they're handled directly, and an array of them will be one solid chunk of memory.
# This makes them 
#If something is not bits, that normally means it 
#Why would we ever want a zero dim array?
function f!(a::Number)
    a = 2.1
end
function f!(a::AbstractArray)
    a[:] = 2.1
end
f!(scalar4double)
scalar4double
f!(zerodim4double)
zerodim4double
f!(dim3array4double)
dim3array4double

# In Julia, variables are not copied.
# The first f! replaced the input number with another number, but the original scalar4double remained bound to 4
# The second f! modified the contents of the arrays
# The arrays stayed the same, but the contents changed.
# Be weary!
zerodim2 = zerodim4double
zerodim2[] = 17.0
zerodim2
zerodim4double
# Julia does not copy unless you ask it to!
# "=" binds a new name to the old variable.
zerodim3 = copy(zerodim2)
zerodim3[] = 9.4
zerodim2
zerodim3

# It is convention in Julia that whenever a function has side effects -- whenever it modifies inputs
# to end that function name with a "bang"!
# The mutated arguments are also placed first.
copy!(zerodim2, zerodim3)
zerodim2
fill!(zerodim2, -1e9)
zerodim4double

# Also, adding commas makes a column vector.
[3f0, 4.0, 2, π]
# For a matrix
[ 1 2 3 ; 4 5 6]
reshape(1:6, 2, 3)
# Note the differences! Order, and
# In Julia, ranges are lazy -- even when reshaped.

# Also, the colon operator also has low priority.
# What is this in R?
# using RCall
3*1:5+7
10:3:22
10:3:22 |> collect

# R is a lazy language, which is great.
# Julia is eager, but many functions and operations are implemented to be lazy.
# Array slices can also be lazy
@views x[4:7]
# On Julia 0.7, transposing matrices is also lazy. Not yet on 0.6.


#If Julia can't promote them all to the same type, it'll give up and create a vector that can hold anything.
[ 42im 9//2 "fish" π ]
#This isn't a good idea. Don't that.


#You can also create undefined arrays. Be cautious, they may contain NaNs, which can infect code if you're not careful!
Array{Float64}(4, 12)

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
#@inline function f(x, y)
function f(x, y)
    out = zero(promote_type(typeof(x), eltype(y)))
    for yᵢ ∈ y
        out += x * yᵢ
    end
    out
end


@code_llvm f(e, x)

#This is a tuple. A tuple's type is partly defined by its length.
@code_llvm f(e, (1.2, -9.5, 3.2, 37.2))
(1.2, -9.5, 3.2, 37.2) |> typeof

@code_llvm f(e, 2.3)
@code_llvm f(2, 3)

function g(f, a, b, c)
    f(a, b) + c
end

# f is automatically inlined at -O3, not at -O2.
g(f, 2, 3, 4)
@code_llvm g(f, 2, 3, 4)

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


# Remember this?
[ 42im 9//2 "fish" π ]
# What if we do want to hold a bunch of different types?
# One straightforward approach is to build our own container!

abstract type four_objects{A,B,C,D} end
struct FourThings{A,B,C,D} <: four_objects{A,B,C,D}
    a::A
    b::B
    c::C
    d::D
end
function FourThings(a::A, b::B, c::C, d::D) where {A,B,C,D}
    FourThings{A,B,C,D}(a,b,c,d)
end
ft1 = FourThings(42im, 9//2, "fish", π)
isbits(ft1)
ft2 = FourThings(42, 9//2,  4f0, π)
isbits(ft2)
mutable struct FourMutableThings{A,B,C,D} <: four_objects{A,B,C,D}
    a::A
    b::B
    c::C
    d::D
end
ft3 = FourMutableThings(42, 9//2,  4f0, π)
isbits(ft3)
struct MyMatrix{T}
    a::T
    b::T
    c::T
    d::T
    e::T
    f::T
    g::T
    h::T
    i::T
    j::T
    k::T
    l::T
    m::T
    n::T
    o::T
    p::T
end
# Then, we get define our operations. For example, multiplication
function Base.:*(x::MyMatrix{T}, y::four) where {T, A,B,C,D,four <: four_objects{A,B,C,D}}
    (
        x.a*y.a + x.e * y.b + x.i * y.c + x.m * y.d,
        x.b*y.a + x.f * y.b + x.j * y.c + x.n * y.d,
        x.c*y.a + x.g * y.b + x.k * y.c + x.o * y.d,
        x.d*y.a + x.h * y.b + x.l * y.c + x.p * y.d,
    )
end
v16 = randn(Float32, 16);
m16 = reshape(v16, 4, 4);
v4 = randn(Float32, 4);
mm = MyMatrix(v16...)
mm * ft2
mm * ft3
ft4 = FourThings(v4...)
ft5 = FourMutableThings(v4...)
m16 * v4
mm * ft4

# What do you expect their performance to be?
# Built in multiplication, vs our own custom version?
using BenchmarkTools
@benchmark $m16 * $v4
@benchmark $mm * $ft2
@benchmark $mm * $ft3
@benchmark $mm * $ft4
@benchmark $mm * $ft5
# Now compare with Julia launched with -O3
# This is why I like -O3

# More cool features
using ForwardDiff

ForwardDiff.derivative(f, 2.3)
ForwardDiff.derivative(sin, 2.3)
cos(2.3)
