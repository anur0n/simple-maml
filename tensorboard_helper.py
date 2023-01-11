from torch.utils.tensorboard import SummaryWriter
import os

class TBHelper:

    def __init__(self, output_folder=None, run_name=None) -> None:
        if output_folder is None:
            output_folder = "output"
        if run_name is None:
            run_name = "cheetah_exp"
        self.writer = SummaryWriter(os.path.join(output_folder, 'plots', run_name))

    def write(self, logs):
        self.writer.add_scalar('Returns/Training Return', logs['train_returns'].sum(), logs['batch'])
        self.writer.add_scalar('Returns/Validation Return', logs['valid_returns'].sum(), logs['batch'])
        self.writer.add_scalar('Loss/loss before', logs['loss_before'].sum(), logs['batch'])
        self.writer.add_scalar('Loss/loss after', logs['loss_after'].sum(), logs['batch'])
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
        else:
            print('Writer is None.')