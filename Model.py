import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MLPProcess import MLPEncoder
from sklearn.neighbors import NearestNeighbors
from Utils import get_mask_from_sequence, get_activation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, BertTokenizer
from VMI import CriticModel, BaselineModel, dv_lower_bound, mine_lower_bound_test, tuba_lower_bound, nwj_lower_bound, infonce_lower_bound, js_fgan_lower_bound, js_lower_bound, smile_lower_bound, interp_lower_bound

def get_output_dim(features_compose_t, features_compose_k, d_out, t_out, k_out):
    if features_compose_k in ['mean', 'sum']:
        classify_dim = d_out
    elif features_compose_k == 'cat':
        classify_dim = d_out * k_out
    else:
        raise NotImplementedError

    if features_compose_t in ['mean', 'sum']:
        classify_dim = classify_dim
    elif features_compose_t == 'cat':
        classify_dim = classify_dim * t_out
    else:
        raise NotImplementedError

    return classify_dim

def mlps_for_cmi(dim, hidden_dim, output_dim, layers, activation, last_acticate):
    '''Return a list of MLPs: [FC + activation]'''
    activation = get_activation(activation)

    seq = [torch.nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [torch.nn.Linear(hidden_dim, hidden_dim), activation()]
    
    if last_acticate=='hardtanh':
        seq += [torch.nn.Linear(hidden_dim, output_dim), nn.Hardtanh(1e-4, 1-1e-4)]
    elif last_acticate=='sigmoid':
        seq += [torch.nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
    else:
        raise NotImplementedError

    return torch.nn.Sequential(*seq)


class MLP_For_CMI(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, layers, activation, last_acticate):
        super().__init__()
        
        activation = get_activation(activation)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, hidden_dim), activation(),
            nn.Linear(hidden_dim, output_dim)
        )
        if last_acticate=='hardtanh':
            self.final_activate = nn.Hardtanh(1e-4, 1-1e-4)
        elif last_acticate=='sigmoid':
            self.final_activate = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, features):

        features = self.mlp(features)

        features = torch.clamp(features, -10, 10)
        features = self.final_activate(features)
 
        return features


def prod_knn_sample(X, Y, Z, batch_size, k_neighbor, radius):
    X, Y, Z = X.detach().cpu().numpy(), Y.detach().cpu().numpy(), Z.detach().cpu().numpy()
    N = X.shape[0]
    # if batch_size % k_neighbor == 0:
    m = batch_size//k_neighbor
    # print(m, N, batch_size)
    index_yz = np.random.choice(range(N), size=m, replace=False)
    neigh = NearestNeighbors(n_neighbors=k_neighbor, radius=radius, metric='euclidean')
    X2 = np.asarray([element for i, element in enumerate(X) if i not in index_yz])
    Z2 = np.asarray([element for i, element in enumerate(Z) if i not in index_yz])
    neigh.fit(Z2)
    Neighbor_indices = neigh.kneighbors(Z[index_yz],return_distance=False)

    index_x = []
    index_y = []
    index_z = []
    for n_i in Neighbor_indices:
        index_x = np.append(index_x, n_i).astype(int)
    for ind in index_yz:
        index_y = np.append(index_y, [ind]*k_neighbor).astype(int)
        index_z = np.append(index_z, [ind]*k_neighbor).astype(int)
                
    batch_x, batch_y, batch_z = torch.tensor(X2[index_x], requires_grad=True), torch.tensor(Y[index_y], requires_grad=True), torch.tensor(Z[index_z], requires_grad=True)
    max_dimension = max(batch_x.shape[1], batch_y.shape[1], batch_z.shape[1])
    if batch_x.shape[1] != max_dimension:
        batch_x = batch_x.repeat(1, max_dimension//batch_x.shape[1])
    if batch_y.shape[1] != max_dimension:
        batch_y = batch_y.repeat(1, max_dimension//batch_y.shape[1])
    if batch_z.shape[1] != max_dimension:
        batch_z = batch_z.repeat(1, max_dimension//batch_z.shape[1])
    # print(batch_x.shape, batch_y.shape, batch_z.shape)
    return batch_x.cuda(), batch_y.cuda(), batch_z.cuda()
    
class VMIEstimator(nn.Module):
    def __init__(self, critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho):
        super().__init__()
        self.critic_type, self.baseline_type, self.bound_type = critic_type, baseline_type, bound_type
        self.critic_model = CriticModel(critic_type, d_common, d_common, hidden_dim=hidden_dim, embed_dim=embed_dim, layers=layers, activation=activation)
        self.baseline_model = BaselineModel(baseline_type, d_common, hidden_dim=hidden_dim, layers=layers, activation=activation, mu=mu, rho=rho)

    def forward(self, features_x, features_y):
        ma_et, ma_rate=1, 0.01
        alpha_logit = 0.01

        scores = self.critic_model(features_x, features_y)

        if self.bound_type == 'mine':
            mi, t, et = mine_lower_bound_test(scores)
            ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
            mi_loss = (torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
        else:
            if self.bound_type == 'dv':
                mi = dv_lower_bound(scores)
            elif self.bound_type == 'tuba':
                log_baseline = self.baseline_model(features_y)
                mi = tuba_lower_bound(scores, log_baseline)
            elif self.bound_type == 'nwj':
                mi = nwj_lower_bound(scores)
            elif self.bound_type == 'infonce':
                mi = infonce_lower_bound(scores)
            elif self.bound_type == 'js':
                mi = js_lower_bound(scores)
            elif self.bound_type == 'js_fgan':
                mi = js_fgan_lower_bound(scores)
            elif self.bound_type == 'smile':
                mi = smile_lower_bound(scores)
            elif self.bound_type == 'interpolate':
                log_baseline = self.baseline_model(features_y)
                mi = interp_lower_bound(scores, log_baseline, alpha_logit)
            else:
                raise NotImplementedError
            mi_loss = -mi

        return mi, mi_loss

class VCMIEstimator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layers, activation, k_neighbor, radius, last_acticate='hardtanh'):
        super().__init__()
        self.classifier = MLP_For_CMI(embed_dim*3, hidden_dim, output_dim=2, layers=layers, activation=activation, last_acticate=last_acticate)
        self.k_neighbor, self.radius = k_neighbor, radius
        self.embed_dim = embed_dim

    def forward(self, features_x, features_y, features_z, random_knn_x, random_knn_y, random_knn_z): # compute I(x;y|z)
        batch_size = features_x.shape[0]
        
        # To same shape
        if features_x.shape[1] != self.embed_dim:
            features_x = features_x.repeat(1, self.embed_dim//features_x.shape[1])
        if features_y.shape[1] != self.embed_dim:
            features_y = features_y.repeat(1, self.embed_dim//features_y.shape[1])
        if features_z.shape[1] != self.embed_dim:
            features_z = features_z.repeat(1, self.embed_dim//features_z.shape[1])
            
        joint_train = torch.cat([features_x, features_y, features_z], dim=1)
        # print(joint_train.shape)
        prod_train = torch.cat([random_knn_x, random_knn_y, random_knn_z], dim=1)
        # print(prod_train.shape)
        # print(joint_train.requires_grad, prod_train.requires_grad)
        if joint_train.shape[0] == prod_train.shape[0]:
            batch_train = torch.cat([joint_train, prod_train], dim=0)

            joint_target = np.repeat([[1,0]],batch_size,axis=0)
            prod_target = np.repeat([[0,1]],batch_size,axis=0)
            target_train = np.concatenate((joint_target,prod_target),axis=0)
            target_train = torch.tensor(target_train).float().cuda()
        else:
            joint_train = joint_train[:prod_train.shape[0]]
            batch_train = torch.cat([joint_train, prod_train], dim=0)
            
            joint_target = np.repeat([[1,0]],prod_train.shape[0],axis=0)
            prod_target = np.repeat([[0,1]],prod_train.shape[0],axis=0)
            target_train = np.concatenate((joint_target,prod_target),axis=0)
            target_train = torch.tensor(target_train).float().cuda()

        out = self.classifier(batch_train)
        # print('Out expected [0:bs] 100%', int(torch.argmax(out, -1).cpu().detach()[:batch_size].sum()), ':', int(torch.argmax(out, -1).cpu().detach()[batch_size:].sum()), ', Acc: ', int(torch.argmax(out, -1).cpu().detach()[batch_size:].sum())/(batch_size*2))
        # print('Out expected [0:bs] 100%',', Acc: ', (torch.argmax(out, -1).detach().cpu() == torch.argmax(target_train, -1).detach().cpu()).sum() / out.shape[0])
        # print('target_train', torch.argmax(target_train, -1).cpu().detach().numpy().tolist())

        # print('out', out.detach().cpu().numpy())
        # print('target_train', target_train.detach().cpu().numpy())
        # out = torch.nan_to_num(torch.clamp(out, min=1e-4, max=1-1e-4), nan=0, posinf=0, neginf=0)
        # assert out.max() <= 1.0 and out.min() >= 0.0, str(out.max()) + str(out.min())
        loss = F.binary_cross_entropy(out, target_train)   
    
        cmi = self.estimate_cmi(batch_train)
        return cmi, loss

    def estimate_cmi(self, batch, cmi_type='nwj'):  # compute I(x;y|z)
        batch_size = batch.shape[0]
        # print(joint_batch.shape, prod_batch.shape)
        gamma = self.classifier(batch)
        # gamma_joint, gamma_prod = self.classifier(joint_batch), self.classifier(prod_batch)
        # print(gamma_joint.shape, gamma_prod.shape)
        gamma_joint = torch.split(gamma, int(batch_size/2), dim=0)[0]
        gamma_joint = torch.split(gamma_joint, 1, dim=1)[0]

        gamma_prod = torch.split(gamma, int(batch_size/2), dim=0)[1]
        gamma_prod = torch.split(gamma_prod, 1, dim=1)[0]
        
        sum1 = torch.sum(torch.log(gamma_joint / (1-gamma_joint + 1e-6)))
        sum2 = torch.sum(torch.log(gamma_prod / (1-gamma_prod + 1e-6)))
                
        if cmi_type == 'nwj':
            cmi = 1.0+(1.0/batch_size)*sum1 - (1.0/batch_size)*sum2
        elif cmi_type == 'dv':
            cmi = (1.0/batch_size)*sum1 - torch.log((1.0/batch_size)*sum2)
        else:
            raise NotImplementedError
        # print(gamma_joint.requires_grad, sum1.requires_grad, cmi.requires_grad)
        return cmi

class Model(nn.Module):
    def __init__(self, opt, d_t, d_a, d_v):
        super(Model, self).__init__()
        d_t, d_a, d_v, d_common, encoders = d_t, d_a, d_v, opt.d_common, opt.encoders
        features_compose_t, features_compose_k, num_class = opt.features_compose_t, opt.features_compose_k, opt.num_class
        self.time_len = opt.time_len
        self.opt = opt

        self.d_t, self.d_a, self.d_v, self.d_common = d_t, d_a, d_v, d_common
        self.encoders = encoders
        assert self.encoders in ['lstm', 'gru', 'conv']
        self.features_compose_t, self.features_compose_k = features_compose_t, features_compose_k
        assert self.features_compose_t in ['mean', 'cat', 'sum']
        assert self.features_compose_k in ['mean', 'cat', 'sum']

        # Bert Extractor
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, local_files_only=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig, local_files_only=True)

        # Extractors
        if self.encoders == 'conv':
            self.conv_a = nn.Conv1d(in_channels=d_a, out_channels=d_common, kernel_size=3, stride=1, padding=1)
            self.conv_v = nn.Conv1d(in_channels=d_v, out_channels=d_common, kernel_size=3, stride=1, padding=1)
        elif self.encoders == 'lstm':
            self.rnn_v = nn.LSTM(d_v, d_common, 1, bidirectional=True, batch_first=True)
            self.rnn_a = nn.LSTM(d_a, d_common, 1, bidirectional=True, batch_first=True)
        elif self.encoders == 'gru':
            self.rnn_v = nn.GRU(d_v, d_common, 2, bidirectional=True, batch_first=True)
            self.rnn_a = nn.GRU(d_a, d_common, 2, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        # LayerNormalize & Dropout
        self.ln_a, self.ln_v = nn.LayerNorm(d_common, eps=1e-6), nn.LayerNorm(d_common, eps=1e-6)
        self.dropout_t, self.dropout_a, self.dropout_v = nn.Dropout(opt.dropout[0]), nn.Dropout(opt.dropout[1]), nn.Dropout(opt.dropout[2])

        # Projector
        self.W_t = nn.Linear(d_t, d_common, bias=False)

        # MLPsEncoder
        self.mlp_encoder = MLPEncoder(activate=opt.activate, d_in=[opt.time_len, 3, d_common], d_hiddens=opt.d_hiddens, d_outs=opt.d_outs, dropouts=opt.dropout_mlp, bias=opt.bias, ln_first=opt.ln_first, res_project=opt.res_project)

        # Define the Classifier
        classify_dim = get_output_dim(self.features_compose_t, self.features_compose_k, opt.d_outs[-1][2], opt.d_outs[-1][0], opt.d_outs[-1][1])
        if classify_dim <= 128:
            self.classifier = nn.Sequential(
                nn.Linear(classify_dim, num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classify_dim, 128),
                nn.ReLU(),
                nn.Dropout(opt.dropout[3]),
                nn.Linear(128, num_class),
            )

        # ===================Define something for estimating VMI========================
        critic_type, baseline_type, bound_type = opt.critic_type, opt.baseline_type, opt.bound_type
        hidden_dim, embed_dim, layers, activation = 256, 128, 2, 'relu'
        mu, rho = 0, 1
        self.critic_type, self.baseline_type, self.bound_type = opt.critic_type, opt.baseline_type, opt.bound_type
        self.k_neighbor, self.radius, self.last_acticate = opt.k_neighbor, opt.radius, opt.cmi_last_acticate
        # Fusion information: I(F;T),I(F;A),I(F;V)
        self.vmi_estimator_f_t = VMIEstimator(critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho)
        self.vmi_estimator_f_a = VMIEstimator(critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho)
        self.vmi_estimator_f_v = VMIEstimator(critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho)
        # Invariant information: I(T;A)+I(T;V)
        self.vmi_estimator_t_a = VMIEstimator(critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho)
        self.vmi_estimator_t_v = VMIEstimator(critic_type, baseline_type, bound_type, d_common, hidden_dim, embed_dim, layers, activation, mu, rho)
        # Specific information: A:I(A;C|T)-I(T;A|C) V:I(V;C|T)-I(T;V|C) T:I(T;C|A)-I(T;A|C)+I(T;C|V)-I(T;V|C)
        # Complementary information: I(T;A|C)+I(T;V|C)
        self.vcmi_estimator_ac_t = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)
        self.vcmi_estimator_ta_c = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)
        self.vcmi_estimator_vc_t = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)
        self.vcmi_estimator_tv_c = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)
        self.vcmi_estimator_tc_a = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)
        self.vcmi_estimator_tc_v = VCMIEstimator(embed_dim, hidden_dim, layers, activation, self.k_neighbor, self.radius, self.last_acticate)

    def compute_vmi_loss_stage1(self, predictions, labels, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):        
        # predictions = predictions.reshape(-1, 1).repeat((1, self.d_common))
        labels = labels.reshape(-1, 1).repeat((1, self.d_common))

        # F_F, T_F, A_F, V_F = F_F.mean(1), T_F.mean(1), A_F.mean(1), V_F.mean(1)
        # print(F_F.shape, T_F.shape, A_F.shape, V_F.shape)

        # Fusion information
        mi_f_t, mi_loss_f_t = self.vmi_estimator_f_t(F_F, T_F)
        mi_f_a, mi_loss_f_a = self.vmi_estimator_f_a(F_F, A_F)
        mi_f_v, mi_loss_f_v = self.vmi_estimator_f_v(F_F, V_F)
        
        # Invariant information
        mi_t_a, mi_loss_t_a = self.vmi_estimator_t_a(T_F, A_F)
        mi_t_v, mi_loss_t_v = self.vmi_estimator_t_v(T_F, V_F)

        # Specific & Complementary information
        batch_size = labels.shape[0]
        random_knn_a, random_knn_c, random_knn_t = prod_knn_sample(A_F_all, C_F_all, T_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_ac_t, cmi_loss_ac_t = self.vcmi_estimator_ac_t(A_F, labels, T_F, random_knn_a, random_knn_c, random_knn_t)

        random_knn_t, random_knn_a, random_knn_c = prod_knn_sample(T_F_all, A_F_all, C_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_ta_c, cmi_loss_ta_c = self.vcmi_estimator_ta_c(T_F, A_F, labels, random_knn_t, random_knn_a, random_knn_c)

        random_knn_v, random_knn_c, random_knn_t = prod_knn_sample(V_F_all, C_F_all, T_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_vc_t, cmi_loss_vc_t = self.vcmi_estimator_vc_t(V_F, labels, T_F, random_knn_v, random_knn_c, random_knn_t)

        random_knn_t, random_knn_v, random_knn_c = prod_knn_sample(T_F_all, V_F_all, C_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tv_c, cmi_loss_tv_c = self.vcmi_estimator_tv_c(T_F, V_F, labels, random_knn_t, random_knn_v, random_knn_c)

        random_knn_t, random_knn_c, random_knn_a = prod_knn_sample(T_F_all, C_F_all, A_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tc_a, cmi_loss_tc_a = self.vcmi_estimator_tc_a(T_F, labels, A_F, random_knn_t, random_knn_c, random_knn_a)

        random_knn_t, random_knn_c, random_knn_v = prod_knn_sample(T_F_all, C_F_all, V_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tc_v, cmi_loss_tc_v = self.vcmi_estimator_tc_v(T_F, labels, V_F, random_knn_t, random_knn_c, random_knn_v)

        return [mi_f_t,mi_f_a,mi_f_v,mi_t_a,mi_t_v,cmi_ac_t,cmi_ta_c,cmi_vc_t,cmi_tv_c,cmi_tc_a,cmi_tc_v], [mi_loss_f_t,mi_loss_f_a,mi_loss_f_v,mi_loss_t_a,mi_loss_t_v,cmi_loss_ac_t,cmi_loss_ta_c,cmi_loss_vc_t,cmi_loss_tv_c,cmi_loss_tc_a,cmi_loss_tc_v]

    def compute_vmi_loss_stage2(self, predictions, labels, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all):        
        # predictions = predictions.reshape(-1, 1).repeat((1, self.d_common))
        labels = labels.reshape(-1, 1).repeat((1, self.d_common))

        # F_F, T_F, A_F, V_F = F_F.mean(1), T_F.mean(1), A_F.mean(1), V_F.mean(1)
        
        # Fusion information
        mi_f_t, mi_loss_f_t = self.vmi_estimator_f_t(F_F, T_F)
        mi_f_a, mi_loss_f_a = self.vmi_estimator_f_a(F_F, A_F)
        mi_f_v, mi_loss_f_v = self.vmi_estimator_f_v(F_F, V_F)
        
        # Invariant information
        mi_t_a, mi_loss_t_a = self.vmi_estimator_t_a(T_F, A_F)
        mi_t_v, mi_loss_t_v = self.vmi_estimator_t_v(T_F, V_F)
        mi_inv = mi_t_a+mi_t_v

        # Specific & Complementary information
        # Specific information: A:I(A;C|T)-I(T;A|C) V:I(V;C|T)-I(T;V|C) T:I(T;C|A)-I(T;A|C)+I(T;C|V)-I(T;V|C)
        # Complementary information: I(T;A|C)+I(T;V|C)
        batch_size = labels.shape[0]
        random_knn_a, random_knn_c, random_knn_t = prod_knn_sample(A_F_all, C_F_all, T_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_ac_t, cmi_loss_ac_t = self.vcmi_estimator_ac_t(A_F, labels, T_F, random_knn_a, random_knn_c, random_knn_t)

        random_knn_t, random_knn_a, random_knn_c = prod_knn_sample(T_F_all, A_F_all, C_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_ta_c, cmi_loss_ta_c = self.vcmi_estimator_ta_c(T_F, A_F, labels, random_knn_t, random_knn_a, random_knn_c)

        random_knn_v, random_knn_c, random_knn_t = prod_knn_sample(V_F_all, C_F_all, T_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_vc_t, cmi_loss_vc_t = self.vcmi_estimator_vc_t(V_F, labels, T_F, random_knn_v, random_knn_c, random_knn_t)

        random_knn_t, random_knn_v, random_knn_c = prod_knn_sample(T_F_all, V_F_all, C_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tv_c, cmi_loss_tv_c = self.vcmi_estimator_tv_c(T_F, V_F, labels, random_knn_t, random_knn_v, random_knn_c)

        random_knn_t, random_knn_c, random_knn_a = prod_knn_sample(T_F_all, C_F_all, A_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tc_a, cmi_loss_tc_a = self.vcmi_estimator_tc_a(T_F, labels, A_F, random_knn_t, random_knn_c, random_knn_a)

        random_knn_t, random_knn_c, random_knn_v = prod_knn_sample(T_F_all, C_F_all, V_F_all, batch_size, self.k_neighbor, self.radius)
        cmi_tc_v, cmi_loss_tc_v = self.vcmi_estimator_tc_v(T_F, labels, V_F, random_knn_t, random_knn_c, random_knn_v)

        mi_spec_t = cmi_tc_a + cmi_tc_v - cmi_ta_c - cmi_tv_c
        mi_spec_a = cmi_ac_t - cmi_ta_c
        mi_spec_v = cmi_vc_t - cmi_tv_c
        mi_comp = cmi_ta_c + cmi_tv_c

        return [mi_f_t,mi_f_a,mi_f_v,mi_inv,mi_spec_t,mi_spec_a,mi_spec_v,mi_comp], [mi_loss_f_t,mi_loss_f_a,mi_loss_f_v,-mi_inv,-mi_spec_t,-mi_spec_a,-mi_spec_v,-mi_comp]

    def forward(self, bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=False, debug=False):

        # Extract Bert features
        t = self.bertmodel(input_ids=bert_sentences, attention_mask=bert_sentence_att_mask, token_type_ids=bert_sentence_types)[0]
        if debug:
            print('Origin:', t.shape, a.shape, v.shape)
        mask_t = bert_sentence_att_mask # Valid = 1
        t = self.W_t(t)
        l_a = a.shape[1]
        l_v = v.shape[1]
        l_t = t.shape[1]

        max_l = max((l_a, l_v, l_t))
        length_padded = max_l
        
        # print(t.shape, a.shape, v.shape)
        # Pad audio & video
        # pad_before = int((length_padded - l_a)/2)
        # pad_after = length_padded - l_a - pad_before
        # a = F.pad(a, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        # a = F.pad(a, (0, 0, 0, length_padded - l_a, 0, 0), "constant", 0)
        # pad_before = int((length_padded - l_v)/2)
        # pad_after = length_padded - l_v - pad_before
        # v = F.pad(v, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        # v = F.pad(v, (0, 0, 0, length_padded - l_v, 0, 0), "constant", 0)
        # pad_before = int((length_padded - l_t)/2)
        # pad_after = length_padded - l_t - pad_before
        # t = F.pad(t, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        # print(t.shape, a.shape, v.shape)
        # t = F.pad(t, (0, 0, 0, length_padded - l_t, 0, 0), "constant", 0)
        # v = F.pad(v, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        # print(a.shape, v.shape, (get_mask_from_sequence(a, dim=-1).shape), get_mask_from_sequence(v, dim=-1).shape)
        # a_fill_pos = (get_mask_from_sequence(a, dim=-1).int() * mask_t).bool()
        # v_fill_pos = (get_mask_from_sequence(v, dim=-1).int() * mask_t).bool()
        # a, v = a.masked_fill(a_fill_pos.unsqueeze(-1), 1e-6), v.masked_fill(v_fill_pos.unsqueeze(-1), 1e-6)
        if debug:
            print('Padded:', t.shape, a.shape, v.shape)
        mask_a = 1 - get_mask_from_sequence(a, dim=-1).int() # Valid = False
        mask_v = 1 - get_mask_from_sequence(v, dim=-1).int() # Valid = False
        if debug:
            print('Padded mask:', mask_t, mask_a, mask_v, sep='\n')
        lengths_a = mask_a.sum(dim=1).cpu()
        lengths_v = mask_v.sum(dim=1).cpu()
        lengths_a[lengths_a==0] = 1
        lengths_v[lengths_v==0] = 1
        # print(lengths_a, lengths_v)
        # l_av_padded = a.shape[1]

        # Extract features
        if self.encoders == 'conv':
            a, v = self.conv_a(a.transpose(1, 2)).transpose(1, 2), self.conv_v(v.transpose(1, 2)).transpose(1, 2)
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))
        elif self.encoders in ['lstm', 'gru']:
            a = pack_padded_sequence(a, lengths_a, batch_first=True, enforce_sorted=False)
            v = pack_padded_sequence(v, lengths_v, batch_first=True, enforce_sorted=False)
            self.rnn_a.flatten_parameters()
            self.rnn_v.flatten_parameters()
            (packed_a, a_out), (packed_v, v_out) = self.rnn_a(a), self.rnn_v(v)
            a, _ = pad_packed_sequence(packed_a, batch_first=True, total_length=l_a)
            v, _ = pad_packed_sequence(packed_v, batch_first=True, total_length=l_v)
            if debug:
                print('After RNN', a.shape, v.shape)
            if self.encoders == 'lstm':
                a_out, v_out =a_out[0], v_out[0]
            a = torch.stack(torch.split(a, self.d_common, dim=-1), -1).sum(-1)
            v = torch.stack(torch.split(v, self.d_common, dim=-1), -1).sum(-1)
            if debug:
                print('After Union', a.shape, v.shape)
            # a, v = F.relu(a), F.relu(v)
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))
            # t = F.relu(self.ln_t(t))
        else:
            raise NotImplementedError
        t, a, v = self.dropout_t(t), self.dropout_a(a), self.dropout_v(v)

        if debug:
            print('After Extracted', t.shape, a.shape, v.shape)

        T_F, A_F, V_F = t.mean(1), a.mean(1), v.mean(1)
        # Padding temporal axis
        t = F.pad(t, (0, 0, 0, self.time_len-t.shape[1], 0, 0), "constant", 0)
        a = F.pad(a, (0, 0, 0, self.time_len-a.shape[1], 0, 0), "constant", 0)
        v = F.pad(v, (0, 0, 0, self.time_len-v.shape[1], 0, 0), "constant", 0)

        # T_F, A_F, V_F = t.mean(1), a.mean(1), v.mean(1)

        # Union 
        x = torch.stack([t, a, v], dim=2)

        if debug:
            print('After Padded and Unioned on Temporal', t.shape, a.shape, v.shape, x.shape)

        # Encoding
        x = self.mlp_encoder(x, mask=None)
        if debug:
            print('After Encoder', x.shape)

        # x_splits = x.split(1, 2)
        # T_F, A_F, V_F = x_splits[0].mean(1).squeeze(), x_splits[1].mean(1).squeeze(),  x_splits[2].mean(1).squeeze()

        # Compose [bs, t, k, d]
        if self.features_compose_k == 'mean':
            fused_features = x.mean(dim=2)
        elif self.features_compose_k == 'sum':
            fused_features = x.sum(dim=2)
        elif self.features_compose_k == 'cat':
            fused_features = torch.cat(torch.split(x, 1, dim=2), dim=-1).squeeze(2)
        else:
            raise NotImplementedError
        # features = fused_features

        if self.features_compose_t == 'mean':
            fused_features = fused_features.mean(dim=1)
        elif self.features_compose_t == 'sum':
            fused_features = fused_features.sum(dim=1)
        elif self.features_compose_t == 'cat':
            fused_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)
        else:
            raise NotImplementedError
        features = fused_features.unsqueeze(1)

        if debug:
            print('Fused', fused_features.shape)
        F_F = features.mean(1)


        # Predictions
        output = self.classifier(fused_features)
        if return_features:
            return [output, F_F, T_F, A_F, V_F]
        else:
            return [output]

def test_cmi_estimator(cmi_type='nwj'):
    joint_batch = F.sigmoid(torch.randn(32, 2))
    prod_batch = F.sigmoid(torch.randn(32, 2))
    batch_size = joint_batch.shape[0]
    gamma_joint = torch.split(joint_batch, 1, dim=1)[0]
    gamma_prod = torch.split(prod_batch, 1, dim=1)[0]
    
    print(gamma_joint.sum(), gamma_prod.sum())
    print((gamma_joint /(1-gamma_joint)).sum(), (gamma_joint /(1-gamma_joint)).sum())
    sum1 = torch.sum(torch.log(gamma_joint / (1-gamma_joint + 1e-6)))
    sum2 = torch.sum(gamma_prod / (1-gamma_prod + 1e-6))
    print(sum1, sum2)
            
    if cmi_type == 'nwj':
        cmi = 1.0+(1.0/batch_size)*sum1 - (1.0/batch_size)*sum2
    elif cmi_type == 'dv':
        cmi = (1.0/batch_size)*sum1 - torch.log((1.0/batch_size)*sum2)
    else:
        raise NotImplementedError

    return cmi

if __name__ == '__main__':
    from Utils import to_gpu, print_gradient

    print('='*40, 'Testing Model', '='*40)
    from types import SimpleNamespace    
    opts = SimpleNamespace(d_t=768, d_a=74, d_v=35, d_common=128, encoders='gru', features_compose_t='mean', features_compose_k='mean', num_class=1, 
            activate='gelu', time_len=50, d_hiddens=[[20, 3, 128],[10, 2, 128]], d_outs=[[20, 3, 128],[10, 2, 128]],
            dropout_mlp=[0.3,0.4,0.5], dropout=[0.3,0.4,0.5,0.6], bias=True, ln_first=False, res_project=[True,True],
            critic_type='separate', baseline_type='constant', bound_type='infonce',
            k_neighbor=2, radius=1.0, cmi_last_acticate='sigmoid',
            )
    # print(opts)

    t = [
        ["And", "the", "very", "very", "last", "one", "one"],
        ["And", "the", "very", "very", "last", "one"],
    ]
    a = torch.randn(2, 7, 74).cuda()
    v = torch.randn(2, 7, 35).cuda()
    c = torch.randn(2, 1).cuda()


    print('='*40, 'Parameters', '='*40)
    model = Model(opts, 768, 74, 35).cuda()
    vmi_params = []
    vcmi_params = []
    main_params = []
    bert_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'bert' in name:
                bert_params.append(name)
            elif 'vmi' in name:
                vmi_params.append(name)
            elif 'vcmi' in name:
                vcmi_params.append(name)
            else: 
                main_params.append(name)
    # print(vmi_params)
    # input()
    # print(vcmi_params)
    # input()
    # print(main_params)
    # input()
    # print(bert_params)
    # input()
    
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only = True)
    sentences = [" ".join(sample) for sample in t]
    bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))
    bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))
    bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))


    print('='*40, 'Model Output', '='*40)
    result = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=True, debug=False)
    print([r.shape for r in result])
    predictions, F_F, T_F, A_F, V_F = result[0], result[1], result[2], result[3], result[4]
    # input()
    
    print('='*40, 'MI calculation', '='*40)
    C_F_all,F_F_all,T_F_all,A_F_all,V_F_all=torch.randn(1000, 1).cuda(),torch.randn(1000, 128).cuda(),torch.randn(1000, 128).cuda(),torch.randn(1000, 128).cuda(),torch.randn(1000, 128).cuda()
    mis, mi_losses = model.compute_vmi_loss_stage1(predictions, c, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
    print(mis, mi_losses)
    mis, mi_losses = model.compute_vmi_loss_stage2(predictions, c, F_F, T_F, A_F, V_F, C_F_all, F_F_all, T_F_all, A_F_all, V_F_all)
    print(mis, mi_losses)
    # input()

    print('='*40, 'Testing gradient', '='*40)
    sumed_losses = result[0].sum() + sum(mi_losses)
    sumed_losses.backward()
    print_gradient(model)
    # for name, p in model.named_parameters():
    #     print(name)