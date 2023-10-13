### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ 6078059a-176d-4e95-92b8-7b39bdeff7cb
begin
	using Pkg
	Pkg.activate("../julia")
	Pkg.add("Ripserer")
	Pkg.add("PlotThemes")
end

# ╔═╡ b3ed9787-f333-4d4a-96e5-56a6541201fe
begin 
	using Ripserer
	using Plots, PlotThemes

	using Images, FileIO
	theme(:juno)
end

# ╔═╡ 10481fbc-6a08-11ee-1c4f-2b0f04f82767
md"""
Setup
"""

# ╔═╡ ec68f1f5-13ad-450f-8bd4-cfb15adb0e6c
md"""
### imports
"""

# ╔═╡ 33fc86a7-f9ef-4ace-8593-54d1a7eeb4b9
md"""
## Persistent homology with Ripser
"""

# ╔═╡ 4ae94cd9-2b69-4e22-8dde-67a1a0cb9b4d
function noisy_circle(n; r1=1, r2=1, noise=0.1)
    points = NTuple{2,Float64}[]
    for _ in 1:n
        θ = 2π * rand()
        point = (
            r1 * sin(θ) + noise * rand() - noise / 2,
            r2 * cos(θ) + noise * rand() - noise / 2,
        )
        push!(points, point)
    end
    return points
end

# ╔═╡ b95d2a4e-b5d8-4ec3-8baa-23bc7db5beea
begin
	points = noisy_circle(100; noise=0)
    result = ripserer(points)

    plt_pts = scatter(
        points;
        legend=false,
        aspect_ratio=1,
        xlim=(-2.2, 2.2),
        ylim=(-2.2, 2.2),
        title="Data",
    )
    plt_diag = plot(result; infinity=3)

    plot(plt_pts, plt_diag; size=(800, 400))
end

# ╔═╡ fa674ef0-aab6-432b-a5ac-4e3bffa37a80
md"""
## COIL20

This dataset contains images of objects rotated by multiples of some angle.
"""

# ╔═╡ 4adadda3-9459-4122-9417-917bd32e73f9
coil_prefix = "../../data/coil-20-unproc"

# ╔═╡ e4467766-25f7-44f7-b028-33ff8aac7306
function load_images(directory_path :: String)
    image_paths = filter(x -> occursin(r"\.(jpg|png|jpeg)$", x), readdir(directory_path))
	
	load_images(image_paths)
end

# ╔═╡ 91aaff71-26c5-42fb-b5b6-48964061756b
function load_images(image_paths :: Vector{String}, dir_path :: String = "")
	[load(joinpath(dir_path, img_file)) for img_file in image_paths]
end

# ╔═╡ 381dd911-ec9a-43ae-a394-560bdfa5cfae


# ╔═╡ c08fb2f0-c253-4cac-9966-de92d77a1b8a
begin	
	images = load_images(["obj1__0.png", "obj1__6.png"], coil_prefix)
	mosaicview(images; nrow=1)
end

# ╔═╡ db3d3288-9505-40f1-8e14-f9d669940c12
typeof(images[1])

# ╔═╡ 6070ff3e-6d8d-48eb-86c6-a96623f74014


# ╔═╡ d2c84a1e-c7ad-4178-a6dd-b59d70385da9
images[1] - 1

# ╔═╡ 162ef0b4-e2be-45a9-b8e1-02b18e6e30db


# ╔═╡ 4873c9f0-0afc-4a0c-919d-25b549081080
ceil((images[1] + images[2]) / 2)

# ╔═╡ e190fb03-4847-46c6-8326-b2c0791c999e


# ╔═╡ cdc1ca21-3568-4f08-b144-cc79a77f72a1


# ╔═╡ Cell order:
# ╠═10481fbc-6a08-11ee-1c4f-2b0f04f82767
# ╠═6078059a-176d-4e95-92b8-7b39bdeff7cb
# ╠═ec68f1f5-13ad-450f-8bd4-cfb15adb0e6c
# ╠═b3ed9787-f333-4d4a-96e5-56a6541201fe
# ╠═33fc86a7-f9ef-4ace-8593-54d1a7eeb4b9
# ╠═4ae94cd9-2b69-4e22-8dde-67a1a0cb9b4d
# ╠═b95d2a4e-b5d8-4ec3-8baa-23bc7db5beea
# ╠═fa674ef0-aab6-432b-a5ac-4e3bffa37a80
# ╠═4adadda3-9459-4122-9417-917bd32e73f9
# ╠═e4467766-25f7-44f7-b028-33ff8aac7306
# ╠═91aaff71-26c5-42fb-b5b6-48964061756b
# ╠═381dd911-ec9a-43ae-a394-560bdfa5cfae
# ╠═c08fb2f0-c253-4cac-9966-de92d77a1b8a
# ╠═db3d3288-9505-40f1-8e14-f9d669940c12
# ╠═6070ff3e-6d8d-48eb-86c6-a96623f74014
# ╠═d2c84a1e-c7ad-4178-a6dd-b59d70385da9
# ╠═162ef0b4-e2be-45a9-b8e1-02b18e6e30db
# ╠═4873c9f0-0afc-4a0c-919d-25b549081080
# ╠═e190fb03-4847-46c6-8326-b2c0791c999e
# ╠═cdc1ca21-3568-4f08-b144-cc79a77f72a1
