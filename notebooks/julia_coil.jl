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

# ╔═╡ fc3b5001-85f9-4494-99fa-df823e184bf2
struct ManifoldLearnerArgs
	method
	dim :: Integer
	k :: Integer
end

# ╔═╡ c50a27e5-d32b-4874-a3b1-f8c6f8434952
struct ManifoldLearner
	reducer
	args :: ManifoldLearnerArgs
	input_data :: Union{Matrix, Nothing}
	reduced_df :: Union{DataFrame, Nothing}
end

# ╔═╡ b957aea4-77db-4b0e-8126-40d683098a21
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ 2557d853-06fb-4ffd-8310-f9fc6abe3f47
begin 
	function make_reduced_dataframe(X_reduced)
		DataFrame(X_reduced, :auto);
	end
	
	function make_reduced_dataframe(X_reduced, class_names)
		reduced_df = DataFrame(X_reduced, :auto);
		reduced_df[!,"class"] = class_names
		reduced_df
	end
end

# ╔═╡ 768f3d06-c291-4543-aa71-0c00b34ae395


# ╔═╡ fb6b14ee-f5fe-4627-a3aa-6cfc27d1f30a


# ╔═╡ 6ca00b8c-47b9-415a-aa11-cb1631ada099
function fit_manifold_learner_impl(method :: Type{UMAP_}, dim, k, data)
	umap = method(data', dim, n_neighbors=k)
	(umap, make_reduced_dataframe(umap.embedding'))
end

# ╔═╡ cbf50c05-5a74-46f7-9af1-82c76005209c
ManifoldLearningType = Union{Type{Isomap}, Type{LTSA}}

# ╔═╡ dad24919-c768-463a-a677-dbc415df0bb5
function fit_manifold_learner_impl(method :: ManifoldLearningType, dim, k, data)
	reducer = fit(method, data'; maxoutdim=dim, k=k)
	(reducer, make_reduced_dataframe(ManifoldLearning.predict(reducer)'))
end

# ╔═╡ d6dda024-56c1-4614-aa7d-ce721e09a0f7
function fit_manifold_learner(args :: ManifoldLearnerArgs, data)
	(reducer, reduced_df) = fit_manifold_learner_impl(args.method, args.dim, args.k, data)
	ManifoldLearner(
		reducer,
		args,
		data,
		reduced_df
	)
end

# ╔═╡ c5b61420-7cfe-4091-82eb-d270c690456a
function add_classes(df, classes)
	df[!, "class"] = classes
	df
end

# ╔═╡ 1d330da1-0acd-49db-bb44-32fe586449f3
md"""
## Isomap
"""

# ╔═╡ 1f06cd86-ac38-465b-8958-d635450be118
begin
	isomap_learner = fit_manifold_learner(ManifoldLearnerArgs(Isomap, 2, 21), coil_matrix)
	add_classes(isomap_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 24b72a9c-9a50-461e-af43-9dd27a9d25d4
md"""

## LTSA
"""

# ╔═╡ 3e4eb31e-8a06-4fcf-ba08-3a20f7dacab7
begin
	ltsa_learner = fit_manifold_learner(ManifoldLearnerArgs(LTSA, 2, 21), coil_matrix)
	add_classes(ltsa_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 929e6c0e-c9c7-4b28-8a94-2937a797421b


# ╔═╡ 5d587c26-f0f3-44c7-8c9d-bc2516be03dd
md"""

## UMAP

"""

# ╔═╡ 3402f6d0-c41a-4a96-a66c-96263ee7defe
begin
	umap_learner = fit_manifold_learner(ManifoldLearnerArgs(UMAP_, 2, 22), coil_matrix)
	add_classes(umap_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
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
# ╠═fc3b5001-85f9-4494-99fa-df823e184bf2
# ╠═c50a27e5-d32b-4874-a3b1-f8c6f8434952
# ╠═b957aea4-77db-4b0e-8126-40d683098a21
# ╠═d6dda024-56c1-4614-aa7d-ce721e09a0f7
# ╠═2557d853-06fb-4ffd-8310-f9fc6abe3f47
# ╠═768f3d06-c291-4543-aa71-0c00b34ae395
# ╠═fb6b14ee-f5fe-4627-a3aa-6cfc27d1f30a
# ╠═6ca00b8c-47b9-415a-aa11-cb1631ada099
# ╠═cbf50c05-5a74-46f7-9af1-82c76005209c
# ╠═dad24919-c768-463a-a677-dbc415df0bb5
# ╠═c5b61420-7cfe-4091-82eb-d270c690456a
# ╠═1d330da1-0acd-49db-bb44-32fe586449f3
# ╠═1f06cd86-ac38-465b-8958-d635450be118
# ╠═24b72a9c-9a50-461e-af43-9dd27a9d25d4
# ╠═3e4eb31e-8a06-4fcf-ba08-3a20f7dacab7
# ╠═929e6c0e-c9c7-4b28-8a94-2937a797421b
# ╠═5d587c26-f0f3-44c7-8c9d-bc2516be03dd
# ╠═3402f6d0-c41a-4a96-a66c-96263ee7defe
