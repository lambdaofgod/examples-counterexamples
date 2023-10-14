### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ e3e020ac-6aaf-11ee-0d9a-85e0beb5ca6e
begin
	using Pkg
	Pkg.activate("../JuliaExamples")
end

# ╔═╡ 70d7489f-45fc-4677-9fbf-3a6916130530
begin
	using JuliaExamples
	using MappedArrays
	using Images
	using MultivariateStats
	using LinearAlgebra
	using Plots
	using PlotThemes

	theme(:juno)
end

# ╔═╡ c7fdaf8f-5d97-4a71-b478-002d60b9e741
md"""
## Setup
"""

# ╔═╡ 59a31fc2-a9d1-4857-b18e-a2591f4327b0
md"""
## Imports
"""

# ╔═╡ 94c2fb01-09db-4a90-b824-28cac32619cf
begin
	function nb_load_image(path)
		load(path)
	end
	function nb_load_image(path, dtype)
		convert(Matrix{dtype}, load(path))
	end
	function nb_load_dataset(root)
	    files = map(x->joinpath(root, x), readdir(root))
	    return mappedarray(p -> nb_load_image(p), files)
	end
end

# ╔═╡ 1d22ad86-87a6-4841-843a-7bc34ebd677f
img_coil_dataset = JuliaExamples.load_dataset("../data/coil-20-unproc");

# ╔═╡ f34f9fac-f9e9-4b07-bacb-2155ac87ac95
begin
	img = img_coil_dataset[1]
	typeof(img)
end

# ╔═╡ 7f901f65-5c90-4485-a7c5-180fe71e9a7e
# ╠═╡ disabled = true
#=╠═╡
function images_dataset_to_vectors_dataset(img_dataset, img_size=(32,32))
	img_flat_size = img_size[1] * img_size[2]
	resized_img_coil_dataset = map(img -> imresize(img, img_size), img_dataset)
	map(img -> real.(reshape(float(img), img_flat_size)), resized_img_coil_dataset)
end
  ╠═╡ =#

# ╔═╡ 2de9492f-424c-478b-918e-b282b5f668fa
begin
	img_size = (64, 64)
	img_flat_size = img_size[1] * img_size[2]
	coil_dataset = images_dataset_to_vectors_dataset(img_coil_dataset);
	coil_matrix = hcat(coil_dataset...)';
end

# ╔═╡ bf2851b6-9d5a-4a02-a3c9-856823322a54
begin
	img_vector = coil_matrix[1,:];
	println(typeof(img_vector))
	println(size(img_vector))
end

# ╔═╡ 4dd21b66-54d7-470e-9080-1ff40628d716
pca = fit(PCA, coil_matrix; maxoutdim=2);

# ╔═╡ 761adf8a-2bc2-4217-9b7c-6483de01b6df
scatter(pca.proj[:,1], pca.proj[:,2])

# ╔═╡ Cell order:
# ╠═c7fdaf8f-5d97-4a71-b478-002d60b9e741
# ╠═e3e020ac-6aaf-11ee-0d9a-85e0beb5ca6e
# ╠═59a31fc2-a9d1-4857-b18e-a2591f4327b0
# ╠═70d7489f-45fc-4677-9fbf-3a6916130530
# ╠═94c2fb01-09db-4a90-b824-28cac32619cf
# ╠═1d22ad86-87a6-4841-843a-7bc34ebd677f
# ╠═f34f9fac-f9e9-4b07-bacb-2155ac87ac95
# ╠═7f901f65-5c90-4485-a7c5-180fe71e9a7e
# ╠═2de9492f-424c-478b-918e-b282b5f668fa
# ╠═bf2851b6-9d5a-4a02-a3c9-856823322a54
# ╠═4dd21b66-54d7-470e-9080-1ff40628d716
# ╠═761adf8a-2bc2-4217-9b7c-6483de01b6df
