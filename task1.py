import pickle


def openDataset(path_file):
    with open(path_file, 'rb') as file:
        dataset = pickle.load(file, encoding='latin1')
    return dataset


train = openDataset('C:\\Users\\orens\\PycharmProjects\\deepLearning_ass1\\cifar-100-python\\train')
