import torch
import torch.optim as optim

class Attacker:
    def __init__(self, model, data, label, criterion, device="cuda", logging=True):
        self.model = model
        self.data = data
        self.label = self.int_to_torch_long(label)
        self.criterion = criterion
        self.device = device
        self.logging = logging
        
        self.model.to(self.device)
        self.model.eval()
    
    def int_to_torch_long(self, label):
        if isinstance(label, int):
            label=torch.Tensor([label]).long()
        return label
    
    def get_data_grad(self, data, label):
        data.requires_grad = True
        pred = self.model(data)
        loss = self.criterion(pred, label)
        self.model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        return data_grad
        
    def check_untargeted_attack(self, perturbed_data):
        with torch.no_grad():
            pred = self.model(perturbed_data.to(self.device))
        pred = pred.argmax().cpu().numpy()
        if self.logging:
            print(f"original {self.label.cpu().numpy()[0]} || attack {pred}               ")
        if self.label.cpu().numpy()[0] != pred:
            return pred
        else:
            return -1
    
    def check_targeted_attack(self, perturbed_data, target):
        with torch.no_grad():
            pred = self.model(perturbed_data.to(self.device))
        pred = pred.argmax().cpu().numpy()
        if self.logging:
            print(f"original {self.label.cpu().numpy()[0]} || attack {pred} || target {target.cpu().numpy()[0]}    ")
        if target.cpu().numpy()[0] == pred:
            return pred
        else:
            return -1
    
    # FGSM # untargeted
    def fgsm_untargeted_attack(self, epsilon):
        data, label = self.data.to(self.device), self.label.to(self.device)
        
        data_grad = self.get_data_grad(data, label)
        perturbed_data = data + epsilon * data_grad.sign() 
        
        k = self.check_untargeted_attack(perturbed_data)
        return perturbed_data.detach().cpu(), k

    # FGSM # target
    def fgsm_targeted_attack(self, epsilon, target):
        data = self.data.to(self.device)
        target = self.int_to_torch_long(target).to(self.device)
        
        data_grad = self.get_data_grad(data, target)
        perturbed_data = data - epsilon * data_grad.sign() 
        k = self.check_targeted_attack(perturbed_data, target)
        return perturbed_data.detach().cpu(), k
    
    # PGD # untargeted
    def pgd_untargeted_attack(self, eps, alpha, PGD_round):
        data, label = self.data.to(self.device), self.label.to(self.device)
        data_raw = data.clone().detach()
        
        data.requires_grad = True
        for i in range(PGD_round):
            if self.logging:
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
            
            data_grad = self.get_data_grad(data, label)
            
            adv_data = data + alpha * data_grad.sign() 
            eta = torch.clamp(adv_data - data_raw.data, min=-eps, max=eps)
            data = (data_raw + eta).detach_()
            
        perturbed_data = data
        k = self.check_untargeted_attack(perturbed_data)
        return perturbed_data.detach().cpu(), k
    
    # PGD # targeted
    def pgd_targeted_attack(self, eps, alpha, PGD_round, target):
        data, target = self.data.to(self.device), self.int_to_torch_long(target).to(self.device)
        data_raw = data.clone().detach()
        
        data.requires_grad = True
        for i in range(PGD_round):
            if self.logging:
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
            
            data_grad = self.get_data_grad(data, target)
            
            adv_data = data - alpha * data_grad.sign() 
            eta = torch.clamp(adv_data - data_raw.data, min=-eps, max=eps)
            data = (data_raw + eta).detach_()
            
        perturbed_data = data
        k = self.check_targeted_attack(perturbed_data, target)
        return perturbed_data.detach().cpu(), k
    
    # MI-FGSM # untargeted
    def mifgsm_untargeted_attack(self, eps, alpha, MIFGSM_round, momentum=0.9):
        data, label = self.data.to(self.device), self.label.to(self.device)
        data_raw = data.clone().detach()
        
        data.requires_grad = True
        data_grad = torch.zeros_like(data).to(self.device)
        for i in range(MIFGSM_round):
            if self.logging:
                print(f"MIFGSM processing ...  {i+1} / {MIFGSM_round}", end="\r")

            data_grad = data_grad * momentum + self.get_data_grad(data, label)
            
            adv_data = data + alpha * data_grad.sign() 
            eta = torch.clamp(adv_data - data_raw.data, min=-eps, max=eps)
            data = (data_raw + eta).detach_()
            
        perturbed_data = data
        k = self.check_untargeted_attack(perturbed_data)
        return perturbed_data.detach().cpu(), k
    
    # MI-FGSM # targeted
    def mifgsm_targeted_attack(self, eps, alpha, MIFGSM_round, target, momentum=0.9):
        data, target = self.data.to(self.device), self.int_to_torch_long(target).to(self.device)
        data_raw = data.clone().detach()
        
        data.requires_grad = True
        data_grad = torch.zeros_like(data).to(self.device)
        for i in range(MIFGSM_round):
            if self.logging:
                print(f"MIFGSM processing ...  {i+1} / {MIFGSM_round}", end="\r")

            data_grad = data_grad * momentum - self.get_data_grad(data, target)
            
            adv_data = data + alpha * data_grad.sign() 
            eta = torch.clamp(adv_data - data_raw.data, min=-eps, max=eps)
            data = (data_raw + eta).detach_()
            
        perturbed_data = data
        k = self.check_targeted_attack(perturbed_data, target)
        return perturbed_data.detach().cpu(), k