using Pkg

Pkg.activate(".")
Pkg.build("JuliaExamples")
Pkg.instantiate()
Pkg.precompile()
