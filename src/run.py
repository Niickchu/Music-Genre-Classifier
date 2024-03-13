def temp_parse():
    import os

    parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    datafolder = parent_folder_path + r"\data\dataset_spec"

    input_data = []
    
    for genre_folder in os.listdir(datafolder):
        genre_folder_path = os.path.join(datafolder, genre_folder)
        for file in os.listdir(genre_folder_path):
            file_path = os.path.join(genre_folder_path, file)
            input_data.append((file_path, genre_folder))

    for data in input_data:
        print(data)



if __name__ == "__main__":
    temp_parse()