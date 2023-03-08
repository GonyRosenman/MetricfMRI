from torch.optim.lr_scheduler import StepLR
import os

class LrHandler():
    def __init__(self,**kwargs):
        self.final_lr = 1e-5 if kwargs.get('final_lr') is None else kwargs.get('final_lr')
        self.base_lr = kwargs.get('lr_init')
        self.gamma = kwargs.get('lr_gamma')
        self.step_size = kwargs.get('lr_step')
        self.task = kwargs.get('task')
        if kwargs.get('loaded_model_weights_path') is None:
            if kwargs.get('explain_task') is None:
                assert kwargs.get('lr_init') is not None

    def set_lr(self,dict_lr):
        if self.base_lr is None:
            self.base_lr = dict_lr
            print('lr message:\nusing last lr of trained model...')
        else:
            assert self.task != 'transformer_reconstruction'

    def set_schedule(self,optimizer):
        self.schedule = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    def schedule_check_and_update(self):
        if self.schedule.get_last_lr()[0] > self.final_lr:
            self.schedule.step()
            if (self.schedule._step_count - 1) % self.step_size == 0:
                print('current lr: {}'.format(self.schedule.get_last_lr()[0]))

class EarlyStop():
    def __init__(self,**kwargs):
        self.is_active = kwargs.get('early_stop')
        self.status = False
        self.report_path = os.path.join(kwargs.get('experiment_folder'),'report.txt')
        if kwargs.get('task').lower() == 'fine_tune':
            self.epsilon = 1e-3
            self.patience = 8
        else:
            self.epsilon = 5e-3
            self.patience = 8

    def condition(self,relevant_history):
        deltas = [(relevant_history[x + 1] - relevant_history[x]) for x in range(len(relevant_history) - 1)]
        deltas_thresholded = [x > self.epsilon for x in deltas]
        if all(deltas_thresholded):
            return True
        elif sum(deltas_thresholded) >= (self.patience-2) and sum(deltas) > self.epsilon*self.patience:
            return True
        else:
            return False

    def check(self,history_writer,epoch):
        history = history_writer.total_val_loss_history
        if len(history) < self.patience or not self.is_active:
            return
        else:
            relevant_history = history[-self.patience:]
            if self.condition(relevant_history):
                self.status = True
                with open(self.report_path,'a+') as f:
                    f.write('stop due to early stopping at epoch {}\n'.format(str(epoch)))
                    f.write(str(relevant_history))
            elif hasattr(history_writer,'train_Balanced_Accuracy_history') and history_writer.train_Balanced_Accuracy_history[-1] > 0.95:
                self.status = True
                with open(self.report_path,'a+') as f:
                    f.write('stop due to early stopping at epoch {}\nwarning: early stopping condition is problematic'.format(str(epoch)))
                    f.write(str(relevant_history))

