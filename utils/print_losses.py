import numpy as np


def print_losses(epoch, iter, iter_per_epoch, losses, print_keys=False):

    if print_keys:
        header_str = 'epoch %d\t\t\tloss\t' % (epoch)

        for key, value in losses.items():
            if key != 'loss':
                if len(key) < 5:
                    key_str = key + ' ' * (5 - len(key))
                    header_str += '\t\t%s' % (key_str)
                else:
                    header_str += '\t\t%s' % (key[0:5])

        print(header_str)

    loss_str = '%05d/%05d: \t%.4f\t' % (iter, iter_per_epoch, np.mean(losses['loss']))

    for key, value in losses.items():
        if key != 'loss':
            loss_str += '\t\t%.4f' % (np.mean(value))

    print(loss_str)
