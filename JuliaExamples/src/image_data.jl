using MappedArrays
using Images
using ImageIO
using FileIO


function load_image(path)
	load(path)
end

function load_image(path, dtype)
	convert(Array{dtype}, load(path))
end

function load_dataset(root)
    files = map(x->joinpath(root, x), readdir(root))
    mappedarray(p -> load_image(p), files)
end

function images_dataset_to_vectors_dataset(img_dataset, img_size=(32,32))
	img_flat_size = img_size[1] * img_size[2]
	resized_img_coil_dataset = mappedarray(img -> imresize(img, img_size), img_dataset)
	mappedarray(img -> real.(reshape(float(img), img_flat_size)), resized_img_coil_dataset)
end
