import time
from options import Options
from dataloader import CreateDataLoader, Dataset
from model.my3rdmodel import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM

def train(opt, data_loader, model, visualizer):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    stage = 0
    model.set_stage(stage)
    for epoch in range(opt.epoch_count, opt.niter*3 + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        tot_errors = None
        avg_errors = None

        for i, data in enumerate(dataset):

            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()
            errors = model.get_current_errors()

            if tot_errors == None:
                tot_errors = errors.copy()
                avg_errors = errors.copy()
            else:
                for k in errors:
                    tot_errors[k] += errors[k]

            # display on visdom
            if total_steps % opt.display_freq == 0:
                results = model.get_current_visuals()
                visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                #errors = model.get_current_errors()
                ttot_errors = tot_errors.copy()

                for k in tot_errors:
                    avg_errors[k] = ttot_errors[k] / (i + 1)

                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, avg_errors, t)
                #if opt.display_id > 0 and total_steps % opt.show_freq == 0:

                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, avg_errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest_'+str(stage))

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest_'+str(stage))
            model.save_networks(str(epoch))

        if (model.stage == 0):
            model.scheduler.step(model.loss_l1_depth)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter,
               time.time() - epoch_start_time))

        if epoch == opt.niter*(stage+1):
            stage = stage+1
            model.set_stage(stage)
            print('end of stage %d', stage)





if __name__ == '__main__':
    opt = Options().parse()
    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    train(opt, data_loader, model, visualizer)