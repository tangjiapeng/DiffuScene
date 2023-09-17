import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.distributions import Normal
import torch.distributed as dist
import math
import numpy as np
import torch.distributed as dist
from tqdm.auto import tqdm
import json
import torch.nn.functional as F
from einops import rearrange, reduce
from .loss import axis_aligned_bbox_overlaps_3d

def norm(v, f):
    v = (v - v.min())/(v.max() - v.min()) - 0.5

    return v, f

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm

def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == 'warm0.1':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.2':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'warm0.5':

        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == 'cosine':

        def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
            """
            Create a beta schedule that discretizes the given alpha_t_bar function,
            which defines the cumulative product of (1-beta) over time from t = [0,1].
            :param num_diffusion_timesteps: the number of betas to produce.
            :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                            produces the cumulative product of (1-beta) up to that
                            part of the diffusion process.
            :param max_beta: the maximum beta to use; use values lower than 1 to
                            prevent singularities.
            """
            betas = []
            for i in range(num_diffusion_timesteps):
                t1 = i / num_diffusion_timesteps
                t2 = (i + 1) / num_diffusion_timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
            
            return np.array(betas).astype(np.float64)
        
        betas_for_alpha_bar(
            time_num,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(schedule_type)
    return betas

'''
models
'''
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + (mean1 - mean2)**2 * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus)*1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min,  torch.ones_like(cdf_min)*1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
    x < 0.001, log_cdf_plus,
    torch.where(x > 0.999, log_one_minus_cdf_min,
             torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta)*1e-12))))
    assert log_probs.shape == x.shape
    return log_probs

class GaussianDiffusion:
    def __init__(self, config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou, train_stats_file):
        # read object property dimension
        self.objectness_dim = config.get("objectness_dim", 1)
        self.class_dim = config.get("class_dim", 21)
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 1)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.objfeat_dim = config.get("objfeat_dim", 0)
        self.loss_separate = loss_separate
        self.loss_iou = loss_iou
        if self.loss_iou:
            with open(train_stats_file, "r") as f:
                train_stats = json.load(f)
            self._centroids = train_stats["bounds_translations"]
            self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
            self._centroids_min, self._centroids_max = torch.from_numpy(self._centroids[0]).float(), torch.from_numpy(self._centroids[1]).float()
            print('load centriods min {} and max {} in Gausssion Diffusion'.format(self._centroids[0], self._centroids[1]))
            
            self._sizes = train_stats["bounds_sizes"]
            self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
            self._sizes_min, self._sizes_max = torch.from_numpy(self._sizes[0]).float(), torch.from_numpy(self._sizes[1]).float()
            print('load sizes min {} and max {} in Gausssion Diffusion'.format( self._sizes[0], self._sizes[1] ))
            
            self._angles = train_stats["bounds_angles"]
            self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self.room_partial_condition = config.get("room_partial_condition", False)
        self.room_arrange_condition = config.get("room_arrange_condition", False)

        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))



    def q_mean_variance(self, x_start, t):  
        """
        diffusion step: q(x_t | x_{t-1})
        """
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)   q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, denoise_fn, data, t, condition, condition_cross, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t, condition, condition_cross)


        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0) 

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        
        elif self.model_mean_type == 'x0':
            x_recon = model_output

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0) 

            eps = self._predict_eps_from_start(data, t=t, x0=x_recon)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)


        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )
    
    def _predict_eps_from_start(self, x_t, t, x0):
        return (
            (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - x0) / \
            self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, condition, condition_cross, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample


    def p_sample_loop(self, denoise_fn, shape, device, condition, condition_cross,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, condition, condition_cross,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs


    '''losses'''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, condition, condition_cross, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=data_start, x_t=data_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None, condition=None, condition_cross=None):
        """
        Training loss calculation
        """
        #B, D, N = data_start.shape
        # make it compatible for 1D 
        if len(data_start.shape) == 3:
            B, D, N = data_start.shape
        elif len(data_start.shape) == 4:
            B, D, M, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            if self.model_mean_type == 'eps':
                target = noise
            elif self.model_mean_type == 'x0':
                target = data_start
            else:
                raise NotImplementedError
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            #eps_recon = denoise_fn(data_t, t, condition, condition_cross)
            denoise_out = denoise_fn(data_t, t, condition, condition_cross)
            assert data_t.shape == data_start.shape
            if len(data_start.shape) == 3:
                assert denoise_out.shape == torch.Size([B, D, N])
            elif len(data_start.shape) == 4:
                assert denoise_out.shape == torch.Size([B, D, M, N])
            assert denoise_out.shape == data_start.shape
            #losses = ((target - denoise_out)**2).mean(dim=list(range(1, len(data_start.shape))))

            if data_start.shape[-1] == self.objectness_dim+self.class_dim+self.bbox_dim+self.objfeat_dim:
                loss_trans = ((target[:, :, 0:self.translation_dim]  - denoise_out[:, :, 0:self.translation_dim])**2).mean(dim=list(range(1, len(data_start.shape))))
                loss_size  = ((target[:, :, self.translation_dim:self.translation_dim+self.size_dim]  - denoise_out[:, :, self.translation_dim:self.translation_dim+self.size_dim])**2).mean(dim=list(range(1, len(data_start.shape))))
                loss_angle = ((target[:, :, self.translation_dim+self.size_dim:self.bbox_dim]  - denoise_out[:, :, self.translation_dim+self.size_dim:self.bbox_dim])**2).mean(dim=list(range(1, len(data_start.shape))))
                loss_bbox  = ((target[:, :, 0:self.bbox_dim]  - denoise_out[:, :, 0:self.bbox_dim])**2).mean(dim=list(range(1, len(data_start.shape))))
                loss_class = ((target[:, :, self.bbox_dim:self.bbox_dim+self.class_dim]  - denoise_out[:, :, self.bbox_dim:self.bbox_dim+self.class_dim])**2).mean(dim=list(range(1, len(data_start.shape))))
                if self.objectness_dim == 0:
                    loss_object = ((target[:, :, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim ] - denoise_out[:, :, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim ])**2).mean(dim=list(range(1, len(data_start.shape))))
                else:
                    loss_object = ((target[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objectness_dim ] - denoise_out[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objectness_dim ])**2).mean(dim=list(range(1, len(data_start.shape))))

                if self.objfeat_dim == 0:
                    loss_objfeat = torch.zeros(B).to(data_start.device)
                else:
                    loss_objfeat =  ((target[:, :, self.bbox_dim+self.class_dim+self.objectness_dim: ] - denoise_out[:, :, self.bbox_dim+self.class_dim+self.objectness_dim: ])**2).mean(dim=list(range(1, len(data_start.shape))))
                    
                    
                if self.loss_separate:
                    losses = loss_bbox + loss_class
                    if self.objectness_dim > 0:
                        losses += loss_object
                    if self.objfeat_dim > 0:
                        losses += loss_objfeat
                else:
                    losses = ((target - denoise_out)**2).mean(dim=list(range(1, len(data_start.shape))))
                    
                if self.loss_iou:
                    # get x_recon & valid mask 
                    if self.model_mean_type == 'eps':
                        x_recon = self._predict_xstart_from_eps(data_t, t, eps=denoise_out)
                    else:
                        x_recon = denoise_out
                    trans_recon = x_recon[:, :, 0:self.translation_dim]
                    sizes_recon = x_recon[:, :, self.translation_dim:self.translation_dim+self.size_dim]
                    if self.objectness_dim >0:
                        obj_recon = x_recon[:, :, self.bbox_dim+self.class_dim:self.bbox_dim+self.class_dim+self.objectness_dim ]
                        valid_mask = (obj_recon >=0).float().squeeze(2)
                    else:
                        obj_recon = x_recon[:, :, self.bbox_dim+self.class_dim-1:self.bbox_dim+self.class_dim]
                        valid_mask = (obj_recon <=0).float().squeeze(2)
                    # descale bounding box to world coordinate system
                    descale_trans = self.descale_to_origin( trans_recon, self._centroids_min.to(data_start.device), self._centroids_max.to(data_start.device) )
                    descale_sizes = self.descale_to_origin( sizes_recon, self._sizes_min.to(data_start.device), self._sizes_max.to(data_start.device) )
                    # get the bbox corners
                    axis_aligned_bbox_corn = torch.cat([ descale_trans - descale_sizes, descale_trans + descale_sizes], dim=-1)
                    assert axis_aligned_bbox_corn.shape[-1] == 6
                    # compute iou
                    bbox_iou = axis_aligned_bbox_overlaps_3d(axis_aligned_bbox_corn, axis_aligned_bbox_corn)
                    bbox_iou_mask = valid_mask[:, :, None] * valid_mask[:, None, :]
                    bbox_iou_valid = bbox_iou * bbox_iou_mask
                    bbox_iou_valid_avg = bbox_iou_valid.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) / ( bbox_iou_mask.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) + 1e-6)
                    # get the iou loss weight w.r.t time
                    w_iou = self._extract(self.alphas_cumprod.to(data_start.device), t, bbox_iou.shape)
                    loss_iou = (w_iou * 0.1 * bbox_iou).mean(dim=list(range(1, len(w_iou.shape))))
                    loss_iou_valid_avg = (w_iou * 0.5 * bbox_iou_valid).sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) / ( bbox_iou_mask.sum( dim=list(range(1, len(bbox_iou_valid.shape))) ) + 1e-6)
                    losses += loss_iou_valid_avg
                else:
                    loss_iou = torch.zeros(B).to(data_start.device)
                    bbox_iou = torch.zeros(B).to(data_start.device)
                    loss_iou_valid_avg = torch.zeros(B).to(data_start.device)
                    bbox_iou_valid_avg = torch.zeros(B).to(data_start.device)
                    
                return losses, {
                    'loss.bbox': loss_bbox.mean(),
                    'loss.trans': loss_trans.mean(),
                    'loss.size': loss_size.mean(),
                    'loss.angle': loss_angle.mean(),
                    'loss.class': loss_class.mean(),
                    'loss.object': loss_object.mean(),
                    'loss.objfeat': loss_objfeat.mean(),
                    'loss.liou': loss_iou_valid_avg.mean(), 
                    'loss.bbox_iou': bbox_iou_valid_avg.mean(), 
                }
            else:
                print('unimplement point dim is: ', data_start.shape[-1])
                raise NotImplementedError
            
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, condition=condition, condition_cross=condition_cross, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses
                   
    
    def descale_to_origin(self, x, minimum, maximum):
        '''
            x shape : BxNx3
            minimum, maximum shape: 3
        '''
        x = (x + 1) / 2
        x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
        return x

    '''debug'''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, condition, condition_cross, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_= torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):

                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn, data_start=x_start, data_t=self.q_sample(x_start=x_start, t=t_b), t=t_b, condition=condition, condition_cross=condition_cross,
                    clip_denoised=clip_denoised, return_pred_xstart=True)
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart-x_start)**2).mean(dim=list(range(1, len(x_start.shape))))
                assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None]==torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()
        


class DiffusionPoint(nn.Module):
    def __init__(self, denoise_net, config, schedule_type='linear', beta_start=0.0001, beta_end=0.02, time_num=1000, 
            loss_type='mse', model_mean_type='eps', model_var_type ='fixedsmall', loss_separate=False, loss_iou=False, train_stats_file=None):
          
        super(DiffusionPoint, self).__init__()
        
        betas = get_betas(schedule_type, beta_start, beta_end, time_num)
        
        self.diffusion = GaussianDiffusion(config, betas, loss_type, model_mean_type, model_var_type, loss_separate, loss_iou, train_stats_file)

        self.model = denoise_net


    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, condition, condition_cross, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt =  self.diffusion.calc_bpd_loop(self._denoise, x0,  condition, condition_cross, clip_denoised)

        return {
            'total_bpd_b': total_bpd_b,
            'terms_bpd': vals_bt,
            'prior_bpd_b': prior_bpd_b,
            'mse_bt':mse_bt
        }


    def _denoise(self, data, t, condition, condition_cross):
        B, D,N= data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t, condition, condition_cross)
        
        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None, condition=None, condition_cross=None):
        
        if len(data.shape) == 3:
            B, D, N = data.shape
        elif len(data.shape) == 4:
            B, D, M, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t!=0] = torch.randn((t!=0).sum(), *noises.shape[1:]).to(noises)

        losses, loss_dict = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises, condition=condition, condition_cross=condition_cross)
        assert losses.shape == t.shape == torch.Size([B])
        return losses.mean(), loss_dict
    

    def gen_samples(self, shape, device, condition=None, condition_cross=None, noise_fn=torch.randn,
                    clip_denoised=True, keep_running=False):
        return self.diffusion.p_sample_loop(self._denoise, shape=shape, device=device, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    def gen_sample_traj(self, shape, device, freq, condition=None, condition_cross=None, noise_fn=torch.randn,
                    clip_denoised=True,keep_running=False):
        return self.diffusion.p_sample_loop_trajectory(self._denoise, shape=shape, device=device, condition=condition, condition_cross=condition_cross, noise_fn=noise_fn, freq=freq,
                                                       clip_denoised=clip_denoised,
                                                       keep_running=keep_running)