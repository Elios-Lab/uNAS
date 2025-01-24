import pickle
import pprint

def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        for index, element in enumerate(data):
            print(f"Element {index + 1}:")
            pprint.pprint(element)
            print("\n")

# Replace 'your_pickle_file.pkl' with the path to your pickle file
read_pickle_file('/home/pigo/uNAS/artifacts/cnn_vwviz/temp_models.pkl')