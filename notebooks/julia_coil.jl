### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ e3e020ac-6aaf-11ee-0d9a-85e0beb5ca6e
begin
	import Pkg
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
	using Ripserer
	using ProgressLogging
	using PersistenceDiagrams
	theme(:juno)
end

# ╔═╡ c7fdaf8f-5d97-4a71-b478-002d60b9e741
md"""
## Setup
"""

# ╔═╡ 5f72520b-017a-4428-836a-b4665e97fa43
2 + 2

# ╔═╡ 943ace02-4e25-4474-95a4-500c796daf3e


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
function images_dataset_to_vectors_dataset(img_dataset, img_size=(32,32))
	img_flat_size = img_size[1] * img_size[2]
	resized_img_coil_dataset = map(img -> imresize(img, img_size), img_dataset)
	map(img -> real.(reshape(float(img), img_flat_size)), resized_img_coil_dataset)
end

# ╔═╡ 2de9492f-424c-478b-918e-b282b5f668fa

begin
	img_size = (64, 64)
	img_flat_size = img_size[1] * img_size[2]
	coil_dataset = images_dataset_to_vectors_dataset(img_coil_dataset);
	raw_coil_matrix = transpose(hcat(coil_dataset...));
	nonconstant_columns = (norm.(raw_coil_matrix |> eachcol)) .> 1e-3;
	coil_matrix = raw_coil_matrix[:, nonconstant_columns];
	#coil_matrix_normalized = normalize(coil_matrix);
end

# ╔═╡ 59f6419d-69d3-40f3-b9a6-8fe6cda2d3f8
begin
	img_vector = coil_matrix[1,:];
	println(typeof(img_vector))
	println(size(img_vector))
end

# ╔═╡ d06e5992-3a35-4d9c-82f8-b94c1abb9042
md"""
## Dimensionality reduction
"""

# ╔═╡ 8f581281-f954-477c-b3ab-237736fee67f
pca = fit(PCA, coil_matrix; maxoutdim=2);

# ╔═╡ 83ff34cc-609b-4628-84f0-ff1ff26e6d9a
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

# ╔═╡ c5b61420-7cfe-4091-82eb-d270c690456a
function add_classes(df, classes)
	df[!, "class"] = classes
	df
end

# ╔═╡ 1d330da1-0acd-49db-bb44-32fe586449f3
md"""
## Isomap
"""

# ╔═╡ 57587a27-19d3-433a-a29f-12b4df2ceb4e
begin
	isomap_learner = fit_manifold_learner(ManifoldLearnerArgs(Isomap, 2, 25), coil_matrix)
	add_classes(isomap_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 24b72a9c-9a50-461e-af43-9dd27a9d25d4
md"""

## LTSA
"""

# ╔═╡ e34f9cba-5234-4c1b-a8d7-78a6c4151540
begin
	ltsa_learner = fit_manifold_learner(ManifoldLearnerArgs(LTSA, 2, 21), coil_matrix)
	add_classes(ltsa_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 5d587c26-f0f3-44c7-8c9d-bc2516be03dd
md"""

## UMAP

"""

# ╔═╡ 0307bc4e-fb3b-4347-9216-3f8745d1229f
begin
	umap_learner = fit_manifold_learner(ManifoldLearnerArgs(UMAP_, 2, 22), coil_matrix)
	add_classes(umap_learner.reduced_df, class_names) |> @vlplot(:point, x=:x1, y=:x2, color="class", width=640, height=480)
end

# ╔═╡ 237f7829-fde9-4b24-b024-45bdddb2a278
md"""

## Comparing the embeddings with topological data analysis

"""

# ╔═╡ 80e6d342-c4d5-4960-a23e-f61c82c8b8ab
function get_ripserer_result(data)
	ripserer(Tuple.(eachrow(data)));
end

# ╔═╡ f892f93c-196a-47ec-b759-6fd33988c131
function get_ripserer_result(manifold_learner :: ManifoldLearner)
	get_ripserer_result(manifold_learner.reduced_df[!,[:x1, :x2]])
end

# ╔═╡ 0f080f44-b4c3-4928-bbe1-8daca9ffca9e
function plot_persistence_diagram(manifold_learner :: ManifoldLearner)
	ripserer_results = get_ripserer_result(manifold_learner)
	plt_diag = plot(ripserer_results; infinity=3, title=string(manifold_learner.args.method))
end

# ╔═╡ 9e2fa136-8db4-4ef0-97e9-a5345ab03571
plot_persistence_diagram(umap_learner)

# ╔═╡ 3522ed36-038c-4f21-ac45-215ba949dfe8
plot_persistence_diagram(isomap_learner)

# ╔═╡ 9fa2e5e9-6f2d-4a53-afac-7b585e658c3c
plot_persistence_diagram(ltsa_learner)

# ╔═╡ c0000587-2a64-4687-976a-67efc806c4a4
md"""
## Comparing stability w.r.t to hyperparameters using TDA
"""

# ╔═╡ 7c36fb88-b740-40df-a889-9bb571f35693
Pkg.instantiate()

# ╔═╡ 513ebd63-98b6-41fc-83b0-d0f0b26ce8e0
using ProgressLogging
using PersistenceDiagrams

# ╔═╡ 17d6e946-893e-4c73-9037-65a1925fa18d
begin
	function manifold_learning_reduce_dim_fn(method)
		(data; k) -> manifold_learning_reduce_dim_impl(method, data; k)
	end

	function manifold_learning_reduce_dim_impl(method, data; k)
		args = ManifoldLearnerArgs(method, 2, k)
		fit_manifold_learner(args, data).reduced_df
	end
	
	function get_wasserstein_distances_for_param_grid(reduce_dim_fn, data, params, diagram_dist=Bottleneck())
		persistence_diagrams = []

		@progress for p in params
			append!(persistence_diagrams, get_ripserer_result(reduce_dim_fn(data;p...)))
		end
		n_params = length(params)
		dist_matrix = zeros((n_params, n_params))
		for i in 1:n_params
			for j in i:n_params
				dist_matrix[i,j] = diagram_dist(persistence_diagrams[i], persistence_diagrams[j])
			end
		end
		dist_matrix
	end
end

# ╔═╡ 87dc1526-c750-49e3-89e2-ac09dc9f4bd1
function flatten_nonzero_values(m)
	m_flat = reshape(m, reduce(*, size(m)))
	[m_v for m_v in m if m_v > 0.0 && isfinite(m_v)]
end

# ╔═╡ 459af5b9-fdff-47da-8b45-57c3f0ff1586
begin
	ks = 5:2:27 |> collect;
	params = [Dict([(:k, k)]) for k in ks];

	method_wasserstein_dists = []
	for method in [Isomap, LTSA, UMAP_]
		reduce_dim_fn = manifold_learning_reduce_dim_fn(method)
		dists = get_wasserstein_distances_for_param_grid(reduce_dim_fn, coil_matrix, params);
		push!(method_wasserstein_dists, (string(method), dists))
	end
end

# ╔═╡ d4642f47-3985-4d00-8f0d-c135c51e212c
for (k,d) in method_wasserstein_dists
	println(k)
	describe(flatten_nonzero_values(d))
end

# ╔═╡ Cell order:
# ╠═c7fdaf8f-5d97-4a71-b478-002d60b9e741
# ╠═e3e020ac-6aaf-11ee-0d9a-85e0beb5ca6e
# ╠═5f72520b-017a-4428-836a-b4665e97fa43
# ╠═943ace02-4e25-4474-95a4-500c796daf3e
# ╠═59a31fc2-a9d1-4857-b18e-a2591f4327b0
# ╠═70d7489f-45fc-4677-9fbf-3a6916130530
# ╠═94c2fb01-09db-4a90-b824-28cac32619cf
# ╠═1d22ad86-87a6-4841-843a-7bc34ebd677f
# ╠═39764d14-cb93-4267-b954-b2a661a38463
# ╠═f34f9fac-f9e9-4b07-bacb-2155ac87ac95
# ╠═f1d371fe-2e0a-423d-9f83-8476a57a44f0
# ╠═7f901f65-5c90-4485-a7c5-180fe71e9a7e
# ╠═2de9492f-424c-478b-918e-b282b5f668fa
# ╠═59f6419d-69d3-40f3-b9a6-8fe6cda2d3f8
# ╠═d06e5992-3a35-4d9c-82f8-b94c1abb9042
# ╠═8f581281-f954-477c-b3ab-237736fee67f
# ╠═83ff34cc-609b-4628-84f0-ff1ff26e6d9a
# ╠═fc3b5001-85f9-4494-99fa-df823e184bf2
# ╠═c50a27e5-d32b-4874-a3b1-f8c6f8434952
# ╠═d6dda024-56c1-4614-aa7d-ce721e09a0f7
# ╠═2557d853-06fb-4ffd-8310-f9fc6abe3f47
# ╠═6ca00b8c-47b9-415a-aa11-cb1631ada099
# ╠═cbf50c05-5a74-46f7-9af1-82c76005209c
# ╠═dad24919-c768-463a-a677-dbc415df0bb5
# ╠═c5b61420-7cfe-4091-82eb-d270c690456a
# ╠═1d330da1-0acd-49db-bb44-32fe586449f3
# ╠═57587a27-19d3-433a-a29f-12b4df2ceb4e
# ╠═24b72a9c-9a50-461e-af43-9dd27a9d25d4
# ╠═e34f9cba-5234-4c1b-a8d7-78a6c4151540
# ╠═5d587c26-f0f3-44c7-8c9d-bc2516be03dd
# ╠═0307bc4e-fb3b-4347-9216-3f8745d1229f
# ╠═237f7829-fde9-4b24-b024-45bdddb2a278
# ╠═80e6d342-c4d5-4960-a23e-f61c82c8b8ab
# ╠═f892f93c-196a-47ec-b759-6fd33988c131
# ╠═0f080f44-b4c3-4928-bbe1-8daca9ffca9e
# ╠═9e2fa136-8db4-4ef0-97e9-a5345ab03571
# ╠═3522ed36-038c-4f21-ac45-215ba949dfe8
# ╠═9fa2e5e9-6f2d-4a53-afac-7b585e658c3c
# ╠═c0000587-2a64-4687-976a-67efc806c4a4
# ╠═7c36fb88-b740-40df-a889-9bb571f35693
# ╠═513ebd63-98b6-41fc-83b0-d0f0b26ce8e0
# ╠═17d6e946-893e-4c73-9037-65a1925fa18d
# ╠═87dc1526-c750-49e3-89e2-ac09dc9f4bd1
# ╠═459af5b9-fdff-47da-8b45-57c3f0ff1586
# ╠═d4642f47-3985-4d00-8f0d-c135c51e212c
