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
	using ManifoldLearning
	using VegaLite
	using DataFrames
	using Statistics
	using UMAP
	
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
begin
	coil_dir = "../data/coil-20-unproc"
	img_coil_dataset = JuliaExamples.load_dataset(coil_dir);
end

# ╔═╡ 39764d14-cb93-4267-b954-b2a661a38463
begin
	class_names = map(p -> split(p, "__")[1], readdir(coil_dir));
	classes = map(p -> UInt8(last(p)), class_names)
end

# ╔═╡ f34f9fac-f9e9-4b07-bacb-2155ac87ac95
begin
	img = img_coil_dataset[1]
	typeof(img)
	img
end

# ╔═╡ f1d371fe-2e0a-423d-9f83-8476a57a44f0
histogram(real.(float(img[:])))

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
	raw_coil_matrix = transpose(hcat(coil_dataset...));
	nonconstant_columns = (norm.(raw_coil_matrix |> eachcol)) .> 1e-3;
	coil_matrix = raw_coil_matrix[:, nonconstant_columns]
	#coil_matrix_normalized = normalize(coil_matrix);
end

# ╔═╡ bf2851b6-9d5a-4a02-a3c9-856823322a54
begin
	img_vector = coil_matrix[1,:];
	println(typeof(img_vector))
	println(size(img_vector))
end

# ╔═╡ d06e5992-3a35-4d9c-82f8-b94c1abb9042
md"""
## Dimensionality reduction
"""

# ╔═╡ 4dd21b66-54d7-470e-9080-1ff40628d716
pca = fit(PCA, coil_matrix; maxoutdim=2);

# ╔═╡ 761adf8a-2bc2-4217-9b7c-6483de01b6df
scatter(pca.proj[:,1], pca.proj[:,2])

# ╔═╡ 554b0697-13bd-43ec-a6a8-0f62b4ca0e74
function reduce_dimension(method, X; maxoutdim=2, k=25)
	reducer = fit(method, X; maxoutdim=maxoutdim, k=k)
	(reducer, ManifoldLearning.predict(reducer))
end

# ╔═╡ e3737480-a3c6-40ba-8c38-de96e2e07a4c
function make_reduced_df(X_reduced)
	DataFrame(X_reduced, :auto);
	reduced_df
end

# ╔═╡ bbdc2431-1252-467c-90a3-5e35cd20ae15
function make_reduced_df(X_reduced, class_names)
	reduced_df = DataFrame(X_reduced, :auto);
	reduced_df[!,"class"] = class_names
	reduced_df
end

# ╔═╡ 968e2464-5def-488b-9557-50412a4fdf49
function get_manifold_learning_results(method, X, class_names; maxoutdim=2, k=25)
	(reducer, X_reduced) = reduce_dimension(method, transpose(X); maxoutdim=maxoutdim, k=k)
	println(X_reduced |> size)
	println(class_names |> size)
	reduced_df = make_reduced_df(transpose(X_reduced), class_names);
	(reducer, reduced_df)
end

# ╔═╡ 1d330da1-0acd-49db-bb44-32fe586449f3
md"""
## Isomap
"""

# ╔═╡ e100ea6f-a28e-44fa-a725-f6b8ee7349a8
begin
	(isomap, coil_isomap_df) = get_manifold_learning_results(Isomap, coil_matrix, class_names; maxoutdim=2, k=21)
	coil_isomap_df |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 24b72a9c-9a50-461e-af43-9dd27a9d25d4
md"""

## LTSA
"""

# ╔═╡ a2e75732-de29-4223-8422-6332315947a4
begin
	(ltsa, coil_ltsa_df) = get_manifold_learning_results(LTSA, coil_matrix, class_names; maxoutdim=2, k=21)
	coil_ltsa_df |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end


# ╔═╡ 5d587c26-f0f3-44c7-8c9d-bc2516be03dd
md"""

## UMAP

"""

# ╔═╡ 3402f6d0-c41a-4a96-a66c-96263ee7defe
begin
	umap = UMAP_(coil_matrix', 2, n_neighbors=21)
	coil_umap_df =  make_reduced_df(umap.embedding', class_names)
	coil_umap_df |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480);
end

# ╔═╡ Cell order:
# ╠═c7fdaf8f-5d97-4a71-b478-002d60b9e741
# ╠═e3e020ac-6aaf-11ee-0d9a-85e0beb5ca6e
# ╠═59a31fc2-a9d1-4857-b18e-a2591f4327b0
# ╠═70d7489f-45fc-4677-9fbf-3a6916130530
# ╠═94c2fb01-09db-4a90-b824-28cac32619cf
# ╠═1d22ad86-87a6-4841-843a-7bc34ebd677f
# ╠═39764d14-cb93-4267-b954-b2a661a38463
# ╠═f34f9fac-f9e9-4b07-bacb-2155ac87ac95
# ╠═f1d371fe-2e0a-423d-9f83-8476a57a44f0
# ╠═7f901f65-5c90-4485-a7c5-180fe71e9a7e
# ╠═2de9492f-424c-478b-918e-b282b5f668fa
# ╠═bf2851b6-9d5a-4a02-a3c9-856823322a54
# ╠═d06e5992-3a35-4d9c-82f8-b94c1abb9042
# ╠═4dd21b66-54d7-470e-9080-1ff40628d716
# ╠═761adf8a-2bc2-4217-9b7c-6483de01b6df
# ╠═554b0697-13bd-43ec-a6a8-0f62b4ca0e74
# ╠═968e2464-5def-488b-9557-50412a4fdf49
# ╠═e3737480-a3c6-40ba-8c38-de96e2e07a4c
# ╠═bbdc2431-1252-467c-90a3-5e35cd20ae15
# ╠═1d330da1-0acd-49db-bb44-32fe586449f3
# ╠═e100ea6f-a28e-44fa-a725-f6b8ee7349a8
# ╠═24b72a9c-9a50-461e-af43-9dd27a9d25d4
# ╠═a2e75732-de29-4223-8422-6332315947a4
# ╠═5d587c26-f0f3-44c7-8c9d-bc2516be03dd
# ╠═3402f6d0-c41a-4a96-a66c-96263ee7defe
