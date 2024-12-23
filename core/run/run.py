import torch, sys, os
from core.utils import tasks, networks, learners, criterions
from core.logger import Logger
from backpack import backpack, extend
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
from core.network.gate import GateLayer, GateLayerGrad
import signal
import traceback
import time
from functools import partial
import wandb
from torch.autograd import grad

def signal_handler(msg, signal, frame):
    print('Exit signal: ', signal)
    cmd, learner = msg
    with open(f'timeout_{learner}.txt', 'a') as f:
        f.write(f"{cmd} \n")
    exit(0)

class Run:
    name = 'run'
    def __init__(self, n_samples=10000, task=None, learner=None, save_path="logs", seed=0, network=None, args=None, **kwargs):
        self.n_samples = int(n_samples)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task = tasks[task]()
        self.task_name = task
        self.learner = learners[learner](networks[network], kwargs)
        self.logger = Logger(save_path)
        self.seed = int(seed)
        self.wdb = True

        if self.wdb:
            self.logrun = wandb.init(
                project="UPDG",
                entity="letuanhf-hanoi-university-of-science-and-technology",
                config=args, 
                name=f"{self.learner}",
                force=True
            )
            
    def compute_param_norms(self, model):
        # Norm-2 của tất cả các tham số trong mô hình (scalar)
        global_norm = torch.sqrt(sum(p.norm(2) ** 2 for p in model.parameters())).item()

        # Norm-2 cho từng lớp (tensor với một giá trị cho mỗi lớp)
        layer_norms = {}
        for i, layer in enumerate(model.children()):
            layer_params = list(layer.parameters())
            if layer_params:  # Bỏ qua các lớp không có tham số
                layer_norm = torch.sqrt(sum(p.norm(2) ** 2 for p in layer_params))
                layer_norms[f'layer_{i}_param'] = layer_norm.item()

        return {"global_param_norm": global_norm, **layer_norms}

    def compute_grads_hess_from_loss(self, model, loss, spcnt=10):
        global_grad_norm = 0.0
        global_hess_norm = 0.0
        layer_grad_norms = {}
        layer_hess_norms = {}

        # Compute gradients
        params = [p for p in model.parameters() if p.requires_grad]
        param_names = [name for name, _ in model.named_parameters()]
        grads = grad(loss, params, retain_graph=True, create_graph=True)

        # Hutchinson Hessian approximation
        hess = [torch.zeros_like(p) for p in params]
        for _ in range(spcnt):
            zs = [torch.randint(0, 2, p.size(), device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution
            h_zs = grad(grads, params, grad_outputs=zs, retain_graph=True, only_inputs=True)
            for idx, (h_z, z) in enumerate(zip(h_zs, zs)):
                hess[idx] += h_z * z / spcnt

        # Compute Norm-2 for gradients and Hessians
        global_grad_norm += sum(g.norm(2) ** 2 for g in grads)
        global_hess_norm += sum(h.norm(2) ** 2 for h in hess)

        for name, g, h in zip(param_names, grads, hess):
            layer_grad_norms[name] = g.norm(2).item()
            layer_hess_norms[name] = h.norm(2).item()

        # Finalize global norms
        global_grad_norm = torch.sqrt(global_grad_norm).item()
        global_hess_norm = torch.sqrt(global_hess_norm).item()

        # Return dictionary
        return {"global_grad_norm": global_grad_norm, **layer_grad_norms}, {"global_hess_norm": global_hess_norm, **layer_hess_norms }

    def compute_activation_norms(self, model, input):
        activation_norms = {}
        hooks = []

        def save_activation_norm(name):
            def hook(module, inp, out):
                norm = out.norm(2).item()
                activation_norms[name] = norm
            return hook

        # Register hooks for each layer
        for name, layer in model.named_modules():
            hooks.append(layer.register_forward_hook(save_activation_norm(name)))

        # Forward pass to compute activations
        with torch.no_grad():
            model(input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activation_norms

    def compute_all_activation_outputs_norms(self, model, input):
        all_activations_norm = 0.0
        hooks = []

        def accumulate_activation_norm(module, inp, out):
            nonlocal all_activations_norm
            all_activations_norm += out.norm(2).item() ** 2

        # Register hooks for each layer
        for layer in model.modules():
            hooks.append(layer.register_forward_hook(accumulate_activation_norm))

        # Forward pass to compute activations
        with torch.no_grad():
            model(input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return torch.sqrt(torch.tensor(all_activations_norm)).item()

    def start(self):
        torch.manual_seed(self.seed)
        losses_per_step_size = []

        if self.task.criterion == 'cross_entropy':
            accuracy_per_step_size = []
        self.learner.set_task(self.task)
        if self.learner.extend:    
            extension = HesScale()
            extension.set_module_extension(GateLayer, GateLayerGrad())
        criterion = extend(criterions[self.task.criterion]()) if self.learner.extend else criterions[self.task.criterion]()
        optimizer = self.learner.optimizer(
            self.learner.parameters, **self.learner.optim_kwargs
        )

        for i in range(self.n_samples):

            # Fetch input and target
            input, target = next(self.task)
            input, target = input.to(self.device), target.to(self.device)

            # Zero gradients, forward, and compute loss
            optimizer.zero_grad()
            output = self.learner.predict(input)
            loss = criterion(output, target)

            # Compute and log parameter norms
            param_norm_dict = self.compute_param_norms(self.learner.network)
            if self.wdb:
                self.logrun.log({f"Param/{k}": v for k, v in param_norm_dict.items()}, step=i)


            # Compute and log gradient and Hessian norms
            grad_dict, hess_dict = self.compute_grads_hess_from_loss(
                model=self.learner.network,
                loss=loss,
                spcnt=50
            )

            if self.wdb:
                self.logrun.log({f"Gradient/{k}": v for k, v in grad_dict.items()}, step=i)
                self.logrun.log({f"Hessian/{k}": v for k, v in hess_dict.items()}, step=i)

            # Compute and log activation norms per layer
            activation_norm_dict = self.compute_activation_norms(self.learner.network, input)
            if self.wdb:
                self.logrun.log({f"activation_norms/{k}": v for k, v in activation_norm_dict.items()}, step=i)

            # Compute and log combined activation norms
            all_activation_norm = self.compute_all_activation_outputs_norms(self.learner.network, input)
            if self.wdb:
                self.logrun.log({"activation_norms/all_activation_norm": all_activation_norm}, step=i)

            # Backpropagation
            if self.learner.extend:
                with backpack(extension):
                    loss.backward()
            else:
                loss.backward()

            # Update parameters
            optimizer.step()

            # Track loss and accuracy
            losses_per_step_size.append(loss.item())
            if self.task.criterion == 'cross_entropy':
                accuracy_per_step_size.append((output.argmax(dim=1) == target).float().mean().item())

        if self.task.criterion == 'cross_entropy':
            self.logger.log(losses=losses_per_step_size,
                            accuracies=accuracy_per_step_size,
                            task=self.task_name, 
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
            )
        else:
            self.logger.log(losses=losses_per_step_size,
                            task=self.task_name,
                            learner=self.learner.name,
                            network=self.learner.network.name,
                            optimizer_hps=self.learner.optim_kwargs,
                            n_samples=self.n_samples,
                            seed=self.seed,
            )


if __name__ == "__main__":
    # Chuyển đối số dòng lệnh thành dictionary
    ll = sys.argv[1:]
    args = {k[2:]: v for k, v in zip(ll[::2], ll[1::2])}
    run = Run(**args, args=args)

    cmd = f"python3 {' '.join(sys.argv)}"
    signal.signal(signal.SIGUSR1, partial(signal_handler, (cmd, args['learner'])))
    current_time = time.time()
    try:
        run.start()
        with open(f"finished_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} time_elapsed: {time.time()-current_time} \n")
    except Exception as e:
        with open(f"failed_{args['learner']}.txt", "a") as f:
            f.write(f"{cmd} \n")
        with open(f"failed_{args['learner']}_msgs.txt", "a") as f:
            f.write(f"{cmd} \n")
            f.write(f"{traceback.format_exc()} \n\n")