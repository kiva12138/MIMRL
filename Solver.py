import os
import pickle

import numpy as np
import torch
from scipy.stats.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

from Config import CUDA
from Customization import (compute_custumized_loss, compute_outputs_from_model, get_customized_loss, other_model_operations)
from DataLoaderLocal import mosi_r2c_7, pom_r2c_7, r2c_2, r2c_7
from DataLoaderUniversal import get_data_loader
from Model import Model
from Utils import (SAM, ccc_loss, SIMSE, FocalLoss,print_gradient, calc_metrics, calc_metrics_pom, ccc_score, log_message, rmse, rmse_score, set_logger, topk_, to_gpu)


class Solver():
    def __init__(self, opt):
        self.opt = opt
        # Get logger and data loader
        self.task_path, self.writer, self.best_valid_model_path, self.best_test_model_path, self.latest_model_path = self.prepare_checkpoint_log()
        log_message(str(self.opt))
        log_message("Making logger and dataset...")
        self.train_loader, self.valid_loader, self.test_loader, self.d_t, self.d_a, self.d_v = get_data_loader(self.opt)
        
        # Get model and optimizer
        log_message("Making model and optimizer...")
        self.model = Model(self.opt, self.d_t, self.d_a, self.d_v)
        other_model_operations(self.model, self.opt)
        self.optimizer_main, self.optimizer_vmi, self.lr_schedule_main, self.lr_schedule_vmi = self.get_optimizer(self.model)
        self.loss_functions = [self.get_task_loss()]
        if opt.parallel:
            log_message("Model paralleling...")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(map(int, CUDA.split(','))))
        self.model = self.model.cuda()

    def solve(self):
        log_message("Start training...")
        # Representing scores, predictions, features of valid, test, test in best valid
        best_score, best_predictions, best_features = [None, None, None], [None, None, None], [None, None, None]
        best_targets = [None, None] # Targets for valid, test
        best_valid_model_state, best_test_model_state = None, None

        C_F_all, F_F_all, T_F_all, A_F_all, V_F_all = [], [], [], [], []
        for epoch in range(self.opt.epochs_num):
            train_loss, train_loss_mi, train_mis, train_score, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all = self.train(epoch, self.train_loader, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
            val_loss, val_mis, val_score, val_predictions, val_targets, val_features = self.evaluate(self.valid_loader, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
            test_loss, test_mis, test_score, test_predictions, test_targets, test_features = self.evaluate(self.test_loader, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)

            # Learning rate schedule
            if self.opt.lr_decrease == 'plateau':
                self.lr_schedule_main.step(val_loss)
                self.lr_schedule_vmi.step(val_loss)
            else:
                self.lr_schedule_main.step()
                self.lr_schedule_vmi.step()
            # Updata metrics, results and features
            if self.current_result_better(best_score[0], val_score):
                best_valid_model_state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optim_main": self.optimizer_main.state_dict(),
                    "optim_vmi": self.optimizer_vmi.state_dict(),
                }
                best_score[0], best_predictions[0], best_features[0] = val_score, val_predictions, val_features
                best_score[2], best_predictions[2], best_features[2] = test_score, test_predictions, test_features
                best_targets[0] = val_targets
                if self.opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_Dec', 'mosei_Dec']:
                    log_message('Better valid score found...')
                    calc_metrics(val_targets, val_predictions)
                    log_message('Test in better valid score found...')
                    calc_metrics(test_targets, test_predictions)
                elif 'pom_SDK' in self.opt.dataset:
                    log_message('Better valid score found...')
                    calc_metrics_pom(val_targets, val_predictions)
                    log_message('Test in better valid score found...')
                    calc_metrics_pom(test_targets, test_predictions)

            if self.current_result_better(best_score[1], test_score):
                best_test_model_state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optim_main": self.optimizer_main.state_dict(),
                    "optim_vmi": self.optimizer_vmi.state_dict(),
                }
                best_score[1], best_predictions[1], best_features[1] = test_score, test_predictions, test_features
                best_targets[1] = test_targets
                log_message('Better test score found...')
                if self.opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_Dec', 'mosei_Dec']:
                    calc_metrics(test_targets, test_predictions)
                elif 'pom_SDK' in self.opt.dataset:
                    calc_metrics_pom(test_targets, test_predictions)

            # best_features[2] = test_features
            # Log the epoch result
            epoch_summary = self.build_message(epoch, train_loss, train_mis, train_score, val_loss,val_mis, val_score, test_loss, test_mis, test_score)
            log_message(epoch_summary)
            self.log_tf_board(epoch, train_loss, train_mis, train_score, val_loss, val_mis, val_score, test_loss, test_mis, test_score)

        # Saving results
        log_message("Training complete.")
        self.writer.close()
        self.log_best_scores(best_score)
        self.save_results(best_predictions, best_targets, best_features, best_valid_model_state, best_test_model_state)

    def prepare_checkpoint_log(self):
        task_path = os.path.join('./TaskRuning', self.opt.task_name)
        best_valid_model_path = os.path.join(task_path, "best_valid_model.pth.tar")
        best_test_model_path = os.path.join(task_path, "best_test_model.pth.tar")
        latest_model_path = os.path.join(task_path, "latest_model.pth.tar")

        os.makedirs(task_path, exist_ok=True)
        set_logger(os.path.join(task_path, "Running.log"))

        writer = SummaryWriter(task_path)
        return task_path, writer, best_valid_model_path, best_test_model_path, latest_model_path
    
    def get_optimizer(self, model):
        vmi_params = []
        main_params = []
        bert_params = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                if 'bert' in name:
                    bert_params.append(p)
                elif 'vmi' in name:
                    vmi_params.append(p)
                elif 'vcmi' in name:
                    vmi_params.append(p)
                else: 
                    main_params.append(p)
            
        main_params_optimized = [
            {'params': bert_params, 'lr': float(self.opt.learning_rate) if self.opt.bert_lr_rate <= 0 else float(self.opt.learning_rate)*self.opt.bert_lr_rate},
            {'params': main_params, 'lr': float(self.opt.learning_rate)},
        ]

        vmi_params_optimized = [
            {'params': vmi_params, 'lr': float(self.opt.learning_rate)*self.opt.mi_lr_rate},
        ]

        if self.opt.optm == "Adam":
            optimizer_main = torch.optim.Adam(main_params_optimized, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay)
            optimizer_vmi = torch.optim.Adam(vmi_params_optimized, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay)
        elif self.opt.optm == "SGD":
            optimizer_main = torch.optim.SGD(main_params_optimized, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay, momentum=0.9 )
            optimizer_vmi = torch.optim.SGD(vmi_params_optimized, lr=float(self.opt.learning_rate), weight_decay=self.opt.weight_decay, momentum=0.9 )
        else:
            raise NotImplementedError

        if self.opt.lr_decrease == 'step':
            self.opt.lr_decrease_iter = int(self.opt.lr_decrease_iter)
            lr_schedule_main = torch.optim.lr_scheduler.StepLR(optimizer_main, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
            lr_schedule_vmi = torch.optim.lr_scheduler.StepLR(optimizer_vmi, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
        elif self.opt.lr_decrease == 'multi_step':
            self.opt.lr_decrease_iter = list((map(int, self.opt.lr_decrease_iter.split('-'))))
            lr_schedule_main = torch.optim.lr_scheduler.MultiStepLR(optimizer_main, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
            lr_schedule_vmi = torch.optim.lr_scheduler.MultiStepLR(optimizer_vmi, self.opt.lr_decrease_iter, self.opt.lr_decrease_rate)
        elif self.opt.lr_decrease == 'exp':
            lr_schedule_main = torch.optim.lr_scheduler.ExponentialLR(optimizer_main, self.opt.lr_decrease_rate)
            lr_schedule_vmi = torch.optim.lr_scheduler.ExponentialLR(optimizer_vmi, self.opt.lr_decrease_rate)
        elif self.opt.lr_decrease == 'plateau':
            mode = 'min' if self.opt.task == 'regression' else 'max'
            lr_schedule_main = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, mode=mode, patience=int(self.opt.lr_decrease_iter), factor=self.opt.lr_decrease_rate,)
            lr_schedule_vmi = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_vmi, mode=mode, patience=int(self.opt.lr_decrease_iter), factor=self.opt.lr_decrease_rate,)
        else:
            raise NotImplementedError
        return optimizer_main, optimizer_vmi, lr_schedule_main, lr_schedule_vmi

    def get_task_loss(self):
        if self.opt.loss == 'Focal':
            loss_func = FocalLoss()
        elif self.opt.loss == 'CE':
            loss_func = torch.nn.CrossEntropyLoss()
        elif self.opt.loss == 'BCE': # 1class or 2class are all OK.
            loss_func = torch.nn.BCEWithLogitsLoss()
        elif self.opt.loss == 'RMSE':
            loss_func = rmse
        elif self.opt.loss == 'MAE':
            loss_func = torch.nn.L1Loss()
        elif self.opt.loss == 'MSE':
            loss_func = torch.nn.MSELoss(reduction='mean')
        elif self.opt.loss == 'SIMSE':
            loss_func = SIMSE()
        elif self.opt.loss == 'CCC':
            loss_func = ccc_loss
        else :
            raise NotImplementedError
            
        return loss_func

    def train(self, epoch, train_loader, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):
        self.model.train()
        running_loss, running_loss_mi, predictions, targets = 0.0, 0.0, [], []
        mis = [0.0 for _ in range(8)]
        
        # Stage1 
        for i in range(self.opt.stage1_n):
            if epoch == 0:
                self.optimizer_vmi.step()
                break
            for _, datas in enumerate(self.train_loader):
                labels = to_gpu(self.get_label_from_datas(datas))
                outputs = compute_outputs_from_model(self.model, datas, self.opt)
                loss, _ = self.compute_loss(outputs, labels, 1, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)

                self.optimizer_vmi.zero_grad()
                loss.backward()
                if self.opt.gradient_clip > 0:
                    torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_clip)
                self.optimizer_vmi.step()
                running_loss_mi += loss.cpu().item()
                if self.opt.check_gradient:
                    print_gradient(self.model)
        
        # Stage2
        C_F_all_new, F_F_all_new, T_F_all_new, A_F_all_new, V_F_all_new = [], [], [], [], []
        for _, datas in enumerate(train_loader):
            labels = to_gpu(self.get_label_from_datas(datas))
            outputs = compute_outputs_from_model(self.model, datas, self.opt)
            C_F_all_new.append(labels.reshape(-1, 1))
            F_F_all_new.append(outputs[1])
            T_F_all_new.append(outputs[2])
            A_F_all_new.append(outputs[3])
            V_F_all_new.append(outputs[4])
            loss, mis_iter = self.compute_loss(outputs, labels, 2, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
            mis = [mis[i] + mis_iter[i].cpu().item() for i in range(len(mis))]

            self.optimizer_main.zero_grad()
            loss.backward()
            if self.opt.gradient_clip > 0:
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.opt.gradient_clip)
            self.optimizer_main.step()
            running_loss += loss.item()
            if self.opt.check_gradient:
                print_gradient(self.model)

            with torch.no_grad():
                predictions += outputs[0].detach().cpu().numpy().tolist()
                targets += labels.detach().cpu().numpy().tolist()

        C_F_all_new, F_F_all_new, T_F_all_new, A_F_all_new, V_F_all_new = torch.cat(C_F_all_new, 0),torch.cat(F_F_all_new, 0),torch.cat(T_F_all_new, 0),torch.cat(A_F_all_new, 0),torch.cat(V_F_all_new, 0)
        predictions, targets = np.array(predictions), np.array(targets)
        train_score = self.get_score_from_result(predictions, targets) # return a dict

        return running_loss/len(train_loader), running_loss_mi/len(train_loader), [mi/len(train_loader) for mi in mis], train_score, C_F_all_new, F_F_all_new, T_F_all_new, A_F_all_new, V_F_all_new

    def evaluate(self, valid_loader, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):
        self.model.eval()
        running_loss, predictions, targets, features = 0.0, [], [], []
        mis = [0.0 for _ in range(8)] # There are 11 metrics we need trace.
        with torch.no_grad():
            for _, datas in enumerate(valid_loader):
                labels = to_gpu(self.get_label_from_datas(datas))
                outputs = compute_outputs_from_model(self.model, datas, self.opt)
                loss, mis_iter = self.compute_loss(outputs, labels, 2, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
                running_loss += loss.item()

                mis = [mis[i] + mis_iter[i].cpu().item() for i in range(len(mis))]

                predictions += outputs[0].cpu().numpy().tolist()
                targets += labels.cpu().numpy().tolist()
                features.append([f.cpu() for f in outputs[1:]])

        predictions, targets = np.array(predictions), np.array(targets)
        valid_score = self.get_score_from_result(predictions, targets)

        return running_loss/len(valid_loader), [mi/len(valid_loader) for mi in mis], valid_score, predictions, targets, features if self.opt.save_best_features else _

    def get_label_from_datas(self, datas):
        if self.opt.dataset in ['mosi_Dec', 'mosei_Dec']:
            labels = datas[5]
            return labels

        labels = datas[3:]

        if self.opt.dataset in ['mosi_SDK', 'mosei_SDK', 'mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:
            if self.opt.task == 'regression':
                labels = labels[0]
            elif self.opt.task == 'classification' and self.opt.num_class==2:
                labels = labels[1]
            elif self.opt.task == 'classification' and self.opt.num_class==7:
                labels = labels[2]
            else:
                raise NotImplementedError
        elif self.opt.dataset == 'pom_SDK':
            if self.opt.task == 'regression':
                labels = labels[0][:, 0]
            elif self.opt.task == 'classification':
                labels = labels[1]
            else:
                raise NotImplementedError
        elif self.opt.dataset in ['youtube', 'youtubev2', 'moud', 'iemocap_20', ]:
            labels = labels[0]
        elif self.opt.dataset in ['mmmo', 'mmmov2']:
            if self.opt.task == 'regression':
                labels = labels[0]
            elif self.opt.task == 'classification':
                labels = labels[1]
            else:
                raise NotImplementedError
        elif self.opt.dataset == 'pom':
            if self.opt.task == 'regression':
                labels = labels[0][:, -3]
            elif self.opt.task == 'classification':
                labels = labels[1]
            else:
                raise NotImplementedError
        elif self.opt.dataset == 'avec2019':
            labels = labels[0]    
        else:
            raise NotImplementedError
        return labels

    def compute_loss(self, outputs, labels, stage, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):
        # Get predictions
        # predictions, T_F_, A_F_, V_F_ = outputs[0], outputs[1], outputs[2] ,outputs[3]
        predictions = outputs[0]

        # Compute task loss from predictions
        task_loss_function = self.loss_functions[0]
        if self.opt.loss in ['Focal', 'CE']:
            predictions, labels = predictions.reshape(-1, self.opt.num_class), labels.reshape(-1,)
            task_loss = task_loss_function(predictions, labels)
        elif self.opt.loss in ['BCE'] and self.opt.num_class==2:
            predictions, labels = predictions.reshape(-1, self.opt.num_class), labels.reshape(-1,)
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.opt.num_class).float()
            task_loss = task_loss_function(predictions, labels_one_hot)
        elif self.opt.loss in ['BCE'] and self.opt.num_class==1:
            predictions, labels = predictions.reshape(-1,), labels.reshape(-1,).float()
            task_loss = task_loss_function(predictions, labels)
        elif self.opt.loss in ['RMSE', 'MAE', 'MSE', 'SIMSE', 'CCC']:
            task_loss = task_loss_function(predictions.reshape(-1, ), labels.reshape(-1, ))
        else:
            raise NotImplementedError

        # Get loss from features
        all_loss, mis = compute_custumized_loss(self.model, task_loss, outputs, labels, self.loss_functions, self.opt, stage, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)

        return all_loss, mis

    def get_score_from_result(self, predictions, targets):
        if self.opt.task == 'classification':
            if self.opt.num_class == 1:
                predictions = np.int64(predictions.reshape(-1,) > 0)
            else:
                _, predictions = topk_(predictions, 1, 1)
            predictions, targets = predictions.reshape(-1,), targets.reshape(-1,)
            acc = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions, average='weighted')
            return {
                str(self.opt.num_class)+'-class_acc': acc,
                str(self.opt.num_class)+'-f1': f1
            }
        elif self.opt.task == 'regression':
            predictions, targets = predictions.reshape(-1,), targets.reshape(-1,)
            mae = mean_absolute_error(targets, predictions)
            corr, _ = pearsonr(predictions, targets )

            if self.opt.dataset in ['mosi_20', 'mosi_50', 'mosei_20', 'mosei_50']:
                if 'mosi' in self.opt.dataset:
                    predictions_7 = [mosi_r2c_7(p) for p in predictions]
                    targets_7 = [mosi_r2c_7(p) for p in targets]
                else:
                    predictions_7 = [r2c_7(p) for p in predictions]
                    targets_7 = [r2c_7(p) for p in targets]

                predictions_2 = [r2c_2(p) for p in predictions]
                targets_2 = [r2c_2(p) for p in targets]
                acc_7 = accuracy_score(targets_7, predictions_7)
                acc_2 = accuracy_score(targets_2, predictions_2)
                f1_2 = f1_score(targets_2, predictions_2, average='weighted')
                f1_7 = f1_score(targets_7, predictions_7, average='weighted')

                return {
                    'mae': mae,
                    'corr': corr,
                    '7-class_acc': acc_7,
                    '2-class_acc': acc_2,
                    '7-f1': f1_7,
                    '2-f1': f1_2,
                }
            elif self.opt.dataset in ['mosi_SDK', 'mosei_SDK','mosi_Dec', 'mosei_Dec']:
                return calc_metrics(targets, predictions, to_print=False)
            elif self.opt.dataset in ['pom_SDK']:
                return calc_metrics_pom(targets, predictions, to_print=False)
            elif self.opt.dataset in ['pom']:
                predictions_7 = [pom_r2c_7(p) for p in predictions]
                targets_7 = [pom_r2c_7(p) for p in targets]
                acc_7 = accuracy_score(targets_7, predictions_7)
                f1_7 = f1_score(targets_7, predictions_7, average='weighted')
                return {
                    'mae': mae,
                    'corr': corr,
                    '7-class_acc': acc_7,
                    '7-f1': f1_7,
                }
            elif self.opt.dataset in ['mmmo', 'mmmov2']:
                predictions_2 = [int(p>=3.5) for p in predictions]
                targets_2 = [int(p>=3.5) for p in targets]
                acc_2 = accuracy_score(targets_2, predictions_2)
                f1_2 = f1_score(targets_2, predictions_2, average='weighted')

                return {
                    'mae': mae,
                    'corr': corr,
                    '2-class_acc': acc_2,
                    '2-f1': f1_2,
                }
            elif self.opt.dataset in ['avec2019']:
                ccc = ccc_score(predictions, targets)
                rmse_ = rmse_score(predictions*25, targets*25)
                return {
                    'mae': mae,
                    'ccc': ccc,
                    'rmse': rmse_,
                }
            else:
                raise NotImplementedError
        else :
            raise NotImplementedError

    def current_result_better(self, best_score, current_score):
        if best_score is None:
            return True
        if self.opt.task == 'classification':
            return current_score[str(self.opt.num_class)+'-class_acc'] > best_score[str(self.opt.num_class)+'-class_acc']
        elif self.opt.task == 'regression':
            if self.opt.dataset != 'avec2019':
                return current_score['mae'] < best_score['mae']
            else:
                return current_score['ccc'] > best_score['ccc']
        else:
            raise NotImplementedError

    def build_message(self, epoch, train_loss, train_mis, train_score, val_loss,val_mis, val_score, test_loss, test_mis, test_score):
        msg = "Epoch:[{:3.0f}]".format(epoch + 1)
        
        msg += " ||"
        msg += " TrainLoss:[{0:.3f}]".format(train_loss)
        msg += " TrainMI_ft/fa/fv/in/st/sa/sv/cp:[{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}]".format(train_mis[0],train_mis[1],train_mis[2],train_mis[3],train_mis[4],train_mis[5],train_mis[6],train_mis[7])
        for key in train_score.keys():
            msg += " Train_"+key+":[{0:6.3f}]".format(train_score[key])

        msg += " ||"
        msg += " ValLoss:[{0:.3f}]".format(val_loss)
        msg += " ValMI_ft/fa/fv/in/st/sa/sv/cp:[{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}]".format(val_mis[0],val_mis[1],val_mis[2],val_mis[3],val_mis[4],val_mis[5],val_mis[6],val_mis[7])
        for key in val_score.keys():
            msg += " Val_"+key+":[{0:6.3f}]".format(val_score[key])

        msg += " ||"
        msg += " TestLoss:[{0:.3f}]".format(test_loss)
        msg += " TestMI_ft/fa/fv/in/st/sa/sv/cp:[{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}/{:.3f}]".format(test_mis[0],test_mis[1],test_mis[2],test_mis[3],test_mis[4],test_mis[5],test_mis[6],test_mis[7])
        for key in test_score.keys():
            msg += " Test_"+key+":[{0:6.3f}]".format(test_score[key])

        return msg

    def build_single_message(self, best_score, mode):
        msg = mode
        for key in best_score.keys():
            msg += " "+key+":[{0:6.3f}]".format(best_score[key])
        return msg

    def log_tf_board(self, epoch, train_loss, train_mis, train_score, val_loss, val_mis, val_score, test_loss, test_mis, test_score):

        self.writer.add_scalar('Train/Loss', train_loss, epoch)
        self.writer.add_scalar('Train/MI_ft', train_mis[0], epoch)
        self.writer.add_scalar('Train/MI_fa', train_mis[1], epoch)
        self.writer.add_scalar('Train/MI_fv', train_mis[2], epoch)
        self.writer.add_scalar('Train/MI_in', train_mis[3], epoch)
        self.writer.add_scalar('Train/MI_spec_t', train_mis[4], epoch)
        self.writer.add_scalar('Train/MI_spec_a', train_mis[5], epoch)
        self.writer.add_scalar('Train/MI_spec_v', train_mis[6], epoch)
        self.writer.add_scalar('Train/MI_comp', train_mis[7], epoch)
        for key in train_score.keys():
            self.writer.add_scalar('Train/'+key, train_score[key], epoch)

        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/MI_ft', val_mis[0], epoch)
        self.writer.add_scalar('Val/MI_fa', val_mis[1], epoch)
        self.writer.add_scalar('Val/MI_fv', val_mis[2], epoch)
        self.writer.add_scalar('Val/MI_in', val_mis[3], epoch)
        self.writer.add_scalar('Val/MI_spec_t', val_mis[4], epoch)
        self.writer.add_scalar('Val/MI_spec_a', val_mis[5], epoch)
        self.writer.add_scalar('Val/MI_spec_v', val_mis[6], epoch)
        self.writer.add_scalar('Val/MI_comp', val_mis[7], epoch)
        for key in val_score.keys():
            self.writer.add_scalar('Val/'+key, val_score[key], epoch)

        self.writer.add_scalar('Test/Loss', test_loss, epoch)
        self.writer.add_scalar('Test/MI_ft', test_mis[0], epoch)
        self.writer.add_scalar('Test/MI_fa', test_mis[1], epoch)
        self.writer.add_scalar('Test/MI_fv', test_mis[2], epoch)
        self.writer.add_scalar('Test/MI_in', test_mis[3], epoch)
        self.writer.add_scalar('Test/MI_spec_t', test_mis[4], epoch)
        self.writer.add_scalar('Test/MI_spec_a', test_mis[5], epoch)
        self.writer.add_scalar('Test/MI_spec_v', test_mis[6], epoch)
        self.writer.add_scalar('Test/MI_comp', test_mis[7], epoch)
        for key in test_score.keys():
            self.writer.add_scalar('Test/'+key, test_score[key], epoch)
        try:
            self.writer.add_scalar('Lr',  self.lr_schedule_main.get_last_lr()[-1], epoch)
        except:
            pass

    def log_best_scores(self, best_score):
        log_message(self.build_single_message(best_score[0], 'Best Valid Score \t\t'))
        log_message(self.build_single_message(best_score[2], 'Test Score at Best Valid \t'))
        log_message(self.build_single_message(best_score[1], 'Best Test Score \t\t'))

    def save_results(self, best_predictions, best_targets, best_features, best_valid_model_state, best_test_model_state):
        np.save(os.path.join(self.task_path, "predictions_val.npy"), best_predictions[0])
        np.save(os.path.join(self.task_path, "predictions_test.npy"), best_predictions[1])
        np.save(os.path.join(self.task_path, "predictions_test_for_valid.npy"), best_predictions[2])

        np.save(os.path.join(self.task_path, "targets_val.npy"), best_targets[0])
        np.save(os.path.join(self.task_path, "targets_test.npy"), best_targets[1])

        if self.opt.save_best_features:
            with open(os.path.join(self.task_path, "features_val.pkl"), 'wb') as f:
                pickle.dump(best_features[0], f)
            with open(os.path.join(self.task_path, "features_test.pkl"), 'wb') as f:
                pickle.dump(best_features[1], f)
            with open(os.path.join(self.task_path, "features_test_for_valid.pkl"), 'wb') as f:
                pickle.dump(best_features[2], f)

        torch.save(best_valid_model_state, self.best_valid_model_path)
        torch.save(best_test_model_state, self.best_test_model_path)
