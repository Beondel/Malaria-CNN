from pathlib import Path
import pickle

def create_datafile():
    image_index = []
    labels = []
    pathlist = Path('./cell_images').glob('**/*.png')
    for path in pathlist:
        image_index.append(str(path))
        if 'Parasitized' in str(path):
            labels.append(1)
        else:
            labels.append(0)
    with open('data.pickle', 'wb') as datafile: 
        pickle.dump(image_index, datafile)
        pickle.dump(labels, datafile)

if __name__ == '__main__':
    create_datafile()