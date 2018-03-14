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
