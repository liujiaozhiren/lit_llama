import datasets

save_path = "../data/where_nyc"
nyc_data = datasets.load_from_disk(save_path)

print("hi")