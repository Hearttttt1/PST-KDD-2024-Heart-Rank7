import torch

# class FastAWP:
#     def __init__(self, model, optimizer, *, adv_param='weight',
#                  adv_lr=0.001, adv_eps=0.001):
#         self.model = model
#         self.optimizer = optimizer
#         self.adv_param = adv_param
#         self.adv_lr = adv_lr
#         self.adv_eps = adv_eps
#         self.backup = {}

#     def perturb(self, inputs, criterion):
#         """
#         Perturb model parameters for AWP gradient
#         Call before loss and loss.backward()
#         """
#         self._save()  # save model parameters
#         self._attack_step()  # perturb weights

#     def _attack_step(self):
#         e = 1e-6
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 grad = self.optimizer.state[param]['exp_avg']
#                 norm_grad = torch.norm(grad)
#                 norm_data = torch.norm(param.detach())

#                 if norm_grad != 0 and not torch.isnan(norm_grad):
#                     # Set lower and upper limit in change
#                     limit_eps = self.adv_eps * param.detach().abs()
#                     param_min = param.data - limit_eps
#                     param_max = param.data + limit_eps

#                     # Perturb along gradient
#                     # w += (adv_lr * |w| / |grad|) * grad
#                     param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

#                     # Apply the limit to the change
#                     param.data.clamp_(param_min, param_max)

#     def _save(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and param.grad is not None and self.adv_param in name:
#                 if name not in self.backup:
#                     self.backup[name] = param.clone().detach()
#                 else:
#                     self.backup[name].copy_(param.data)

#     def restore(self):
#         """
#         Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
#         Call after loss.backward(), before optimizer.step()
#         """
#         for name, param in self.model.named_parameters():
#             if name in self.backup:
#                 param.data.copy_(self.backup[name])


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler


    def attack_backward(self, inputs):
        # if (self.adv_lr == 0) or (epoch < self.start_epoch):
        #     return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            outputs_adv = self.model(inputs)
            adv_loss, tr_logits = outputs_adv['loss'], outputs_adv['logits']
            self.optimizer.zero_grad()
            adv_loss.backward()
            
        self._restore()


    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])


    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )


    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}