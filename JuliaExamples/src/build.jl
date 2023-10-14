using Pkg

Pkg.activate(".")
Pkg.precompile()
Pkg.build()
Pkg.use("JuliaExamples")
