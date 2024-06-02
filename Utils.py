import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_squared_error
from torch.autograd import Function


def print_gradient(model):
    for name, parms in model.named_parameters():
        if 'bert' in name:
            continue
        print('-->name:', name)
        print('-->para:', parms.sum())
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad.sum())
        print("="*25)


def ccc_loss(output, target):
    output, target = output.reshape(-1,), target.reshape(-1,)
    out_mean = torch.mean(output) # 0?
    target_mean = torch.mean(target)

    covariance = torch.mean( (output - out_mean) * (target - target_mean) ) # 0?
    target_var = torch.mean( (target - target_mean)**2)
    out_var = torch.mean( (output - out_mean)**2 ) # 0?

    ccc = 2.0 * covariance/(target_var + out_var + (target_mean-out_mean)**2 + 1e-10) # 0?
    loss_ccc = 1.0 - ccc

    return loss_ccc


def ccc_score(x, y):
    x, y = x.reshape(-1,), y.reshape(-1,)
    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = np.nanmean((x - x_mean) ** 2)
    y_var = np.nanmean((y - y_mean) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return CCC


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def log_message(message):
    # Logging some message after defining logging and path
    logging.log(msg=message, level=logging.DEBUG)


def get_activation(activation):
    activation_dict = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "hardshrink": nn.Hardshrink,
        "hardtanh": nn.Hardtanh,
        "leakyrelu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "relu": nn.ReLU,
        "rrelu": nn.RReLU,
        "tanh": nn.Tanh,
    }
    return activation_dict[activation]


def get_activation_function(activation):
    activation_dict = {
        "elu": F.elu,
        "gelu": F.gelu,
        "hardshrink": F.hardshrink,
        "hardtanh": F.hardtanh,
        "leakyrelu": F.leaky_relu,
        "prelu": F.prelu,
        "relu": F.relu,
        "rrelu": F.rrelu,
        "tanh": torch.tanh,
    }
    return activation_dict[activation]


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def get_seperate_acc(labels, predictions, num_class):
    accs = [0 for i in range(num_class)]
    alls = [0 for i in range(num_class)]
    corrects = [0 for i in range(num_class)]
    for label, prediction in zip(labels, predictions):
        alls[label] += 1
        if label == prediction:
            corrects[label] += 1
    for i in range(num_class):
        accs[i] = '{0:5.1f}%'.format(100 * corrects[i] / alls[i])
    return ','.join(accs)


# For mosi mosei
def calc_metrics(y_true, y_pred, to_print=True):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """
    y_true, y_pred = y_true.reshape(-1,), y_pred.reshape(-1,)

    test_preds, test_truth = y_pred, y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    # print(test_preds.shape, test_truth.shape)

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truthstest_truth
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    # pos - neg
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2_pos_neg = accuracy_score(binary_truth, binary_preds)
    f1_pos_neg = f1_score(binary_truth, binary_preds, average='weighted')
    if to_print:
        logging.log(msg="MAE: "+str(mae), level=logging.DEBUG)
        logging.log(msg="Corr: "+str(corr), level=logging.DEBUG)
        logging.log(msg="Acc5: "+str(mult_a5), level=logging.DEBUG)
        logging.log(msg="Acc7: "+str(mult_a7), level=logging.DEBUG)
        logging.log(msg="Acc2 (pos/neg): "+str(acc2_pos_neg), level=logging.DEBUG)
        logging.log(msg="F1 (pos/neg): "+str(f1_pos_neg), level=logging.DEBUG)
        logging.log(msg="Classification Report (pos/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    # non-neg - neg
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    acc2_nonneg_neg = accuracy_score(binary_truth, binary_preds)
    f1_nonneg_neg = f1_score(binary_truth, binary_preds, average='weighted')
    if to_print:
        logging.log(msg="Acc2 (non-neg/neg): " +str(acc2_nonneg_neg), level=logging.DEBUG)
        logging.log(msg="F1 (non-neg/neg): " +str(f1_nonneg_neg), level=logging.DEBUG)
        logging.log(msg="Classification Report (non-neg/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    return {
        'mae': mae,
        'corr': corr,
        '7-class_acc': mult_a7,
        '5-class_acc': mult_a5,
        '2(pos/neg)-class_acc': acc2_pos_neg,
        '2(nneg/neg)-class_acc': acc2_nonneg_neg,
        '2(pos/neg)-class_f1': f1_pos_neg,
        '2(nneg/neg)-class_f1': f1_nonneg_neg,
    }


def calc_metrics_pom(y_true, y_pred, to_print=True):
    """
    Metric scheme adapted from:
    https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    """

    test_preds = y_pred
    test_truth = y_true

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    # pos - neg
    binary_truth = test_truth[non_zeros] > 0
    binary_preds = test_preds[non_zeros] > 0
    acc2_pos_neg = accuracy_score(binary_truth, binary_preds)
    f1_pos_neg = f1_score(binary_truth, binary_preds, average='weighted')

    if to_print:
        logging.log(msg="MAE: "+str(mae), level=logging.DEBUG)
        logging.log(msg="Corr: "+str(corr), level=logging.DEBUG)
        logging.log(msg="Acc2 (pos/neg): "+str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)
        logging.log(msg="Classification Report (pos/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    # non-neg - neg
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0
    acc2_nonneg_neg = accuracy_score(binary_truth, binary_preds)
    f1_nonneg_neg = f1_score(binary_truth, binary_preds, average='weighted')

    if to_print:
        logging.log(msg="Acc2 (non-neg/neg): " +str(accuracy_score(binary_truth, binary_preds)), level=logging.DEBUG)
        logging.log(msg="Classification Report (non-neg/neg): ", level=logging.DEBUG)
        logging.log(msg=classification_report(binary_truth, binary_preds, digits=5), level=logging.DEBUG)

    return {
        'mae': mae,
        'corr': corr,
        '2(pos/neg)-class_acc': acc2_pos_neg,
        '2(nneg/neg)-class_acc': acc2_nonneg_neg,
        '2(pos/neg)-class_f1': f1_pos_neg,
        '2(nneg/neg)-class_f1': f1_nonneg_neg,
    }


def str2listoffints(v):
    temp_list = v.split('=')
    temp_list = [list(map(int, t.split("-"))) for t in temp_list]
    return temp_list


def str2bool(v):
    """string to boolean"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError("Boolean value expected." + v)


def str2bools(v):
    return list(map(str2bool, v.split("-")))


def str2floats(v):
    return list(map(float, v.split("-")))


def whether_type_str(data):
    return "str" in str(type(data))


def get_predictions_tensor(predictions):
    pred_vals, pred_indices = torch.max(predictions, dim=-1)
    return pred_indices


def showImageNormalized(data):
    # 0.5 0.5 norm
    import matplotlib.pyplot as plt

    data = data.numpy().transpose((1, 2, 0))
    data = data / 2 + 0.5
    plt.imshow(data)
    plt.show()


def rmse(output, target):
    output, target = output.reshape(-1,), target.reshape(
        -1,
    )
    rmse_loss = ((output - target) ** 2).mean()**0.5
    return rmse_loss


def rmse_score(output, target):
    return np.sqrt(mean_squared_error(output, target))


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


def get_mask_from_sequence(sequence, dim):
    return torch.sum(torch.abs(sequence), dim=dim) == 0


def lock_all_params(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False
    return model


def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x


def to_cpu(x):
    """Variable => Tensor"""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data


def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort, topk_index_sort


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


class PadCollateForSequence:
    def __init__(self, dim=0, pad_tensor_pos=[2, 3], data_kind=4):
        self.dim = dim
        self.pad_tensor_pos = pad_tensor_pos
        self.data_kind = data_kind

    def pad_collate(self, batch):
        new_batch = []

        for pos in range(self.data_kind):
            if pos not in self.pad_tensor_pos:
                if not isinstance(batch[0][pos], torch.Tensor):
                    new_batch.append(torch.Tensor([x[pos] for x in batch]))
                else:
                    new_batch.append(torch.stack([x[pos] for x in batch]), dim=0)
            else:
                max_len = max(map(lambda x: x[pos].shape[self.dim], batch))
                padded = list(
                    map(lambda x: pad_tensor(x[pos], pad=max_len, dim=self.dim), batch)
                )
                padded = torch.stack(padded, dim=0)
                new_batch.append(padded)

        return new_batch

    def __call__(self, batch):
        return self.pad_collate(batch)


class CMD(torch.nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class DiffLoss(torch.nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(torch.nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


def aug_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    std_features = torch.std(data, dim=aug_dim)
    max_features, _ = torch.max(data, dim=aug_dim)
    min_features, _ = torch.min(data, dim=aug_dim)
    union_feature = torch.cat(
        (mean_features, std_features, min_features, max_features), dim=-1
    )
    return union_feature


def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features


class UnivariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(UnivariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, var_1, var_2):  # var is variance
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = var_1.type(dtype=torch.float64)  # sigma_1 := sigma_nat^2
        sigma_2 = var_2.type(dtype=torch.float64)  # sigma_2 := sigma_adv^2

        # log(sqrt(sigma2)/sqrt(sigma1))
        term_1 = (sigma_2.sqrt() / sigma_1.sqrt()).log()

        # (sigma_1 + (mu1-mu2)^2)/(2*sigma_2)
        term_2 = (sigma_1 + (mu1 - mu2).pow(2)) / (2 * sigma_2)

        # Calc kl divergence on entire batch
        kl = term_1 + term_2 - 0.5

        # Calculate mean kl_d loss
        if self.reduction == "mean":
            kl_agg = torch.mean(kl)
        elif self.reduction == "sum":
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(
                f"Reduction type not implemented: {self.reduction}"
            )

        return kl_agg


class MultVariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(MultVariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, mu1, mu2, var_1, var_2):  # var is standard deviation
        mu1, mu2 = mu1.type(dtype=torch.float64), mu2.type(dtype=torch.float64)
        sigma_1 = var_1.type(dtype=torch.float64)
        sigma_2 = var_2.type(dtype=torch.float64)

        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal(
            (torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1
        ).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(
            torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
            (mu2 - mu1).unsqueeze(-1),
        ).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5 * (term_1 - n + term_2 + term_3)

        # Calculate mean kl_d loss
        if self.reduction == "mean":
            kl_agg = torch.mean(kl)
        elif self.reduction == "sum":
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(
                f"Reduction type not implemented: {self.reduction}"
            )

        return kl_agg


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-1 * logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


if __name__ == "__main__":
    preds = np.array([0.1*i for i in range(100)])
    labels = np.array([1 - 0.2*i for i in range(100)])
    results = calc_metrics(labels, preds, to_print=False)
    print(results)
