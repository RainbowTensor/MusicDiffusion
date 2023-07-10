import torch


class Diffuzz():
    def __init__(self, s=0.008, device="cpu"):
        self.device = device
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def _alpha_cumprod(self, t):
        alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod
        return alpha_cumprod.clamp(0.0001, 0.9999) 

    def diffuse(self, x, t, noise=None): # t -> [0, 1]
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1-alpha_cumprod).sqrt() * noise, noise

    def undiffuse(self, x, t, t_prev, noise, sampler=None):
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, noise)
        
    def sample(self, model, model_inputs, shape, t_start=1.0, t_end=0.0, timesteps=20, x_init=None, cfg=3.0, unconditional_inputs=None, sampler='ddim'):
        r_range = torch.linspace(t_start, t_end, timesteps+2)[1:][:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)
        # --- select the sampler
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self)
            else:
                raise ValueError(f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler =  sampler(self)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
        # ---  
        preds = []
        x = sampler.init_x(shape) if x_init is None else x_init.clone()
        for i in range(0, timesteps):
            pred_noise = model(x, r_range[i], **model_inputs)
            if cfg is not None:
                if unconditional_inputs is None:
                    unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
                pred_noise_unconditional = model(x, r_range[i], **unconditional_inputs)
                pred_noise = torch.lerp(pred_noise_unconditional, pred_noise, cfg)
            x = self.undiffuse(x, r_range[i], r_range[i+1], pred_noise, sampler=sampler)
            preds.append(x)
        return preds
        
    def p2_weight(self, t, k=1.0, gamma=1.0):
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma

class SimpleSampler():
    def __init__(self, diffuzz):
        self.current_step = -1
        self.diffuzz = diffuzz

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape, device=self.diffuzz.device)

    def step(self, x, t, t_prev, noise):
        raise NotImplementedError("You should override the 'apply' function.")

class DDPMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        alpha = (alpha_cumprod / alpha_cumprod_prev)

        mu = (1.0 / alpha).sqrt() * (x - (1-alpha) * noise / (1-alpha_cumprod).sqrt())
        std = ((1-alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * torch.randn_like(mu)
        return mu + std * (t_prev != 0).float().view(t_prev.size(0), *[1 for _ in x.shape[1:]])

class DDIMSampler(SimpleSampler):
    def step(self, x, t, t_prev, noise):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise

sampler_dict = {
    'ddpm': DDPMSampler,
    'ddim': DDIMSampler,
}