import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter


from Utils import get_activation, to_gpu


def mlps(dim, hidden_dim, output_dim, layers, activation):
    '''Return a list of MLPs: [FC + activation]'''
    activation = get_activation(activation)

    seq = [torch.nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [torch.nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [torch.nn.Linear(hidden_dim, output_dim)]

    return torch.nn.Sequential(*seq)


class CriticModel(torch.nn.Module): 
    '''
    Separate critic: space: 2*batch_size
    Concat critic: space: batch_size**2
    Input features should be [batch_size, dim]
    Each output element i,j is a scalar in R. f(xi,yj)
    '''
    def __init__(self, critic_type, dim_x, dim_y, hidden_dim=256, embed_dim=128, layers=2, activation='relu'):
        super(CriticModel, self).__init__()
        self.critic_type = critic_type
        if critic_type == 'separate':
            self.MLP_g = mlps(dim_x, hidden_dim, embed_dim, layers, activation)
            self.MLP_h = mlps(dim_y, hidden_dim, embed_dim, layers, activation)
            self.init_mlp_params(self.MLP_g)
            self.init_mlp_params(self.MLP_h)
        elif critic_type == 'concat':
            print('If using concat critic for MINE, embed_dim will take no effect.')
            self.MLP_f = mlps(dim_x + dim_y, hidden_dim, 1, layers, activation)
            self.init_mlp_params(self.MLP_f)
        else:
            raise NotImplementedError

    def init_mlp_params(self, mlps):
        for layer in mlps:
            if 'Linear' in type(layer).__name__:
                # torch.nn.init.normal_(layer.weight,std=1)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x, y):
        if self.critic_type == 'separate':
            x_ = self.MLP_g(x)
            y_ = self.MLP_h(y)
            scores = torch.matmul(y_, torch.transpose(x_, 0, 1))
        elif self.critic_type == 'concat':
            batch_size = x.shape[0]
            # Tile all possible combinations of x and y
            x_tiled, y_tiled = torch.stack([x] * batch_size, dim=0), torch.stack([y] * batch_size, dim=1)
            xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size * batch_size, -1]) # [bs * bs, x_dim + y_dim]
            # Compute scores for each x_i, y_j pair.
            scores = self.MLP_f(xy_pairs)
            scores = torch.reshape(scores, [batch_size, batch_size]).t()
        else:
            raise NotImplementedError

        return scores


class BaselineModel(torch.nn.Module): 
    '''
    Gaussain: non parameters
    Constant: non parameters
    Unnormalized: trainable
    Input features should be [batch_size, dim]
    Output a log-baseline scalar: log a(y)
    '''
    def __init__(self, baseline_type, dim_y, hidden_dim=256, layers=2, activation='relu', mu=0, rho=1):
        super(BaselineModel, self).__init__()
        self.baseline_type = baseline_type
        if baseline_type == 'unnormalized':
            self.MLP = mlps(dim_y, hidden_dim, 1, layers, activation)            
            self.init_mlp_params(self.MLP)
        elif baseline_type == 'constant':
            pass
        elif baseline_type == 'gaussain':
            self.gaussain_dist = torch.distributions.Normal(mu, rho)
        else:
            raise NotImplementedError
            
    def init_mlp_params(self, mlps):
        for layer in mlps:
            if 'Linear' in type(layer).__name__:
                # torch.nn.init.normal_(layer.weight,std=1)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, y):
        batch_size = y.shape[0]
        if self.baseline_type == 'unnormalized':
            result = self.MLP(y).reshape(batch_size, 1)
        elif self.baseline_type == 'constant':
            result = to_gpu(torch.zeros(size=(batch_size, 1)))
        elif self.baseline_type == 'gaussain':
            result = to_gpu(torch.sum(self.gaussain_dist.log_prob(y), -1).reshape(batch_size, 1))
        else:
            raise NotImplementedError

        return result


def logmeanexp_diag(x):
    """Tool function: Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)
    logsumexp = torch.logsumexp(x.diag(), dim=0)
    num_elem = batch_size
    return logsumexp - torch.log(torch.tensor(num_elem).float())


def logmeanexp_nodiag(x):
    """Tool function: Compute logmeanexp over the non-diagonal elements of x."""
    batch_size = x.size(0)
    logsumexp = torch.logsumexp(x - to_gpu(torch.diag(torch.inf * torch.ones(batch_size))), dim=(0, 1))
    num_elem = batch_size * (batch_size - 1.)
    return logsumexp - torch.log(torch.tensor(num_elem))


def exp_nodiag(x):
    """Tool function: Compute exp over the non-diagonal elements of x."""
    batch_size = x.size(0)
    exp = torch.exp(x - to_gpu(torch.diag(torch.inf * torch.ones(batch_size))))
    return exp


def dv_lower_bound(scores):
    first_term = scores.diag().mean()
    second_term = logmeanexp_nodiag(scores)
    return first_term - second_term


def mine_lower_bound_test(scores):
    first_term = scores.diag().mean()
    second_term = logmeanexp_nodiag(scores)
    return first_term - second_term, scores.diag(), exp_nodiag(scores)


def tuba_lower_bound(scores, log_baseline=None):
    # if baseline= 1 then equal to NWJ
    if log_baseline is not None:
        scores = scores - log_baseline
    joint_term = scores.diag().mean()  # First term is an expectation over samples from the joint, which are the diagonal elmements of the scores matrix.
    marg_term = torch.exp(logmeanexp_nodiag(scores)) # Second term is an expectation over samples from the marginal, which are the off-diagonal elements of the scores matrix.
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(scores):
    nll = torch.mean(scores.diag() - torch.logsumexp(scores, dim=1))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    # mi = np.log(scores.size(0)) + nll
    return mi


def js_fgan_lower_bound(scores):
    batch_size = scores.size(0)
    f_diag = scores.diag()
    first_term = torch.mean(-F.softplus(-f_diag))
    second_term = (torch.sum(F.softplus(scores)) - torch.sum(F.softplus(f_diag))) / (batch_size * (batch_size - 1))
    return first_term - second_term


def js_lower_bound(scores):
    nwj = nwj_lower_bound(scores)
    js = js_fgan_lower_bound(scores)
    with torch.no_grad():
        nwj_js = nwj - js
    return js + nwj_js.detach()


def smile_lower_bound(scores, clip=None):
    clip = 1
    if clip is not None:
        f_ = torch.clamp(scores, -clip, clip)
    else:
        f_ = scores
    z = logmeanexp_nodiag(f_)
    dv = scores.diag().mean() - z

    js = js_fgan_lower_bound(scores)

    with torch.no_grad():
        dv_js = dv - js
    return js + dv_js.detach()


def log_interpolate(log_a, log_b, alpha_logit:float):
    '''
    Numerically stable implmentation of log(alpha * a + (1-alpha) *b)
    Compute the log baseline for the interpolated bound baseline is a(y)
    '''
    alpha_logit = float(alpha_logit)
    log_alpha = -F.softplus(torch.tensor(-alpha_logit))
    log_1_minus_alpha = -F.softplus(torch.tensor(alpha_logit))
    y = torch.logsumexp( torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b)), dim=0)
    return y


def compute_log_loomean(scores):
    '''Compute the log leave one out mean of the exponentiated scores'''
    max_scores, _ = torch.max(scores, dim=1,keepdim=True)

    lse_minus_max = torch.logsumexp(scores-max_scores,dim=1,keepdim=True)
    d = lse_minus_max + (max_scores - scores)
    
    d_not_ok = torch.eq(d, 0.)
    d_ok = ~d_not_ok
    safe_d = torch.where(d_ok, d, torch.ones_like(d)) #Replace zeros by 1 in d

    loo_lse = scores + (safe_d + torch.log(-torch.expm1(-safe_d))) #Stable implementation of softplus_inverse
    loo_lme = loo_lse - np.log(scores.size()[1] - 1.)
    return loo_lme


def interp_lower_bound(scores, baseline, alpha_logit):
    '''
    New lower bound on mutual information proposed by Ben Poole and al.
    in "On Variational Bounds of Mutual Information"
    It allows to explictily control the biais-variance trade-off.
    For MI estimation -> This bound with a small alpha is much more stable but
    still small biais than NWJ / Mine-f bound !
    Return a scalar, the lower bound on MI
    '''
    batch_size = scores.size()[0]
    nce_baseline = compute_log_loomean(scores)

    interpolated_baseline = log_interpolate(nce_baseline, baseline.repeat(1,batch_size), alpha_logit) #Interpolate NCE baseline with a learnt baseline

    #Marginal distribution term
    critic_marg = scores - torch.diag(interpolated_baseline)
    marg_term = torch.exp(logmeanexp_nodiag(critic_marg))

    #Joint distribution term
    critic_joint = torch.diag(scores) - interpolated_baseline
    joint_term = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / (batch_size * (batch_size - 1.))
    return 1 + joint_term - marg_term


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_MINE(critic_model, baseline_model, bound_type, xy_loader, epochs, lr=5e-4, alpha_logit=0.0, log=False, ma_et=1, ma_rate=0.01, weight_decay=0.999):
    if baseline_model.baseline_type == 'unnormalized':
        optimizer = torch.optim.Adamax(list(critic_model.parameters())+list(baseline_model.parameters()),lr=lr)
        emas = [EMA(critic_model, weight_decay), EMA(baseline_model, weight_decay)]
    else:
        optimizer = torch.optim.Adamax(critic_model.parameters(), lr=lr)
        emas = [EMA(critic_model, weight_decay)]
    for ema in emas:
        ema.register()

    if bound_type=='interpolated':
        assert baseline_model.baseline_type!='constant', "If using Interpolate bound, baseline should not be none!"    

    history_mi = []
    ma_et, ma_rate = ma_et, ma_rate
    for epoch in range(epochs):
        mi_epoch = 0
        for _, features in enumerate(xy_loader):
            features_x, features_y = to_gpu(features[0]), to_gpu(features[1])
            scores = critic_model(features_x, features_y)

            if bound_type == 'mine':
                mi, t, et = mine_lower_bound_test(scores)
                ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
                mi_loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
            else:
                if bound_type == 'dv':
                    mi = dv_lower_bound(scores)
                elif bound_type == 'tuba':
                    log_baseline = baseline_model(features_y)
                    mi = tuba_lower_bound(scores, log_baseline)
                elif bound_type == 'nwj':
                    mi = nwj_lower_bound(scores)
                elif bound_type == 'infonce':
                    mi = infonce_lower_bound(scores)
                elif bound_type == 'js':
                    mi = js_lower_bound(scores)
                elif bound_type == 'js_fgan':
                    mi = js_fgan_lower_bound(scores)
                elif bound_type == 'smile':
                    mi = smile_lower_bound(scores)
                elif bound_type == 'interpolate':
                    log_baseline = baseline_model(features_y)
                    mi = interp_lower_bound(scores, log_baseline, alpha_logit)
                else:
                    raise NotImplementedError
                mi_loss = -mi

            optimizer.zero_grad()
            mi_loss.backward()
            optimizer.step()
            for ema in emas:
                ema.update()
                ema.apply_shadow()

            mi_epoch += mi.detach().cpu().numpy()
        mi_epoch = mi_epoch / len(xy_loader)
        if log and epoch % 50 == 0:
            print('Epoch', epoch, ':', np.round(mi_epoch, 3))
        history_mi.append(mi_epoch)
    return np.asarray(history_mi)


def compute_MI(critic_type, baseline_type, bound_type, features_x, features_y, dim_x, dim_y,
                hidden_dim=256, embed_dim=128, layers=2, activation='relu', mu=0, rho=1,
                epochs=100, batch_size=128, lr=5e-4, alpha_logit=0.0, log=False, ma_et=1, ma_rate=0.01, weight_decay=0.999,
                estimation='mean'):
    '''
    critic_type: separate concat
    baseline_type: constant gaussain unnormalized
    bound_type: dv smile tuba nwj infonce js js_fgan smile interpolate
    features_x, features_y: list of [bs, dim]
    For representation learning purpose, use infoNCE lower bound
    For MI estimation purpose, use interpolated bound with a low alpha
    '''
    critic_model = to_gpu(CriticModel(critic_type, dim_x, dim_y, hidden_dim=hidden_dim, embed_dim=embed_dim, layers=layers, activation=activation))
    baseline_model = to_gpu(BaselineModel(baseline_type, dim_y, hidden_dim=hidden_dim, layers=layers, activation=activation, mu=mu, rho=rho))
    xy_loader = DataLoader(TensorDataset(features_x.clone().detach(), features_y.clone().detach()), batch_size=batch_size)

    history_mi = train_MINE(critic_model, baseline_model, bound_type, xy_loader, epochs, lr, alpha_logit, log, ma_et, ma_rate, weight_decay=weight_decay)
    del critic_model, baseline_model, xy_loader
    if estimation == 'max':
        mi_score = np.max(history_mi)
    elif estimation == 'mean':
        mi_score = np.mean(history_mi[-50:-1])
    elif estimation == 'smooth':
        history_mi = savgol_filter(history_mi, 51, 3)
        mi_score = np.mean(history_mi[-50:-1])
    else:
        raise NotImplementedError

    return mi_score, history_mi


def show_history_mi(history_mi, mi_score, true_mi):
    plt.plot(history_mi)
    plt.hlines(mi_score, 0, len(history_mi))
    plt.text(10, mi_score+np.max(history_mi)/50, str(np.round(mi_score,2)))

    plt.title(f"Mutual information estimation, true MI is "+str(np.round(true_mi,2)))
    plt.show()

def sample_correlated_gaussian(rho=0.5, dim=20, num_samples=1000):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.split(torch.normal(0, 1, size=(num_samples, 2 * dim)), dim, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2, dtype=torch.float32)) * eps
    return x, y

def rho_to_mi(dim, rho):
    return -0.5  * np.log(1-rho**2) * dim


##################################################
#-------------------CMI realted------------------#
##################################################





# Log(BS) This is the upper bound of InfoNCE, and log(BS/a) is the upper bound of interpolate bound
# If I > log(BS), the bound is loose
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    exit(0)

    history_mi = []
    for i in range(20, 300, 20):
        dim, rho = i, 0.7
        features_x, features_y = sample_correlated_gaussian(dim=dim, rho=rho, num_samples=2000)
        print('Features samples:', features_x.shape, features_y.shape)    
        print('True MI:', np.round(rho_to_mi(dim, rho),2))
        # mi_estimated = pyMIestimator(features_x, features_y, k=3, base=np.exp(1))
        # dv smile tuba nwj infonce js js_fgan smile interpolate
        mi_estimated, _ = compute_MI(critic_type='concat', baseline_type='constant', bound_type='interpolate', # interpolate
                                    features_x=features_x, features_y=features_x, dim_x=dim, dim_y=dim,
                                    hidden_dim=256, embed_dim=128, layers=2, activation='relu', mu=0, rho=1,
                                    epochs=2, batch_size=4, lr=5e-4, alpha_logit=0.1, log=True, weight_decay=0.999,
                                    estimation='max')
        history_mi.append(mi_estimated)
    
    plt.plot(history_mi)
    plt.show()
    # print('Traditional estimated MI:', pyMIestimator(features_x, features_y, k=5, base=np.exp(1)))
    
    # mi_score, history_mi = compute_MI(critic_type='separate', baseline_type='constant', bound_type='smile', # interpolate
    #                                     features_x=features_x, features_y=features_x, dim_x=dim, dim_y=dim,
    #                                     hidden_dim=256, embed_dim=128, layers=2, activation='relu', mu=0, rho=1,
    #                                     epochs=200, batch_size=512, lr=5e-4, alpha_logit=0.1, log=True, weight_decay=0.999,
    #                                     estimation='smooth')
    #                                     # dv tuba nwj infonce js js_fgan smile interpolate
    # print('Estimated MI:', mi_score)
    # show_history_mi(history_mi, mi_score, rho_to_mi(dim, rho))

    # mis = []
    # for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #     features_x, features_y = sample_correlated_gaussian(dim=dim, rho=rho, num_samples=2000)
    #     mi_score, history_mi = compute_MI(critic_type='separate', baseline_type='constant', bound_type='smile', # interpolate
    #                                     features_x=features_x, features_y=features_x, dim_x=dim, dim_y=dim,
    #                                     hidden_dim=256, embed_dim=128, layers=2, activation='relu', mu=0, rho=1,
    #                                     epochs=200, batch_size=512, lr=5e-4, alpha_logit=0.1, log=False, weight_decay=0.999,
    #                                     estimation='max')
    #     mis.append(mi_score)
    #     print(mi_score)
    # plt.plot([0.1, 0.3, 0.5, 0.7, 0.9], mis)
    # plt.show()
