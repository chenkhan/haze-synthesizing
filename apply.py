from dataloader import TestDataset,CreateDataLoader
from options import TestOptions
from model.my3rdmodel import create_model

if __name__ == '__main__':
    opt = TestOptions().parse()
    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#haze-free images = %d' % dataset_size)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        model.save_result(opt.result_dir)
