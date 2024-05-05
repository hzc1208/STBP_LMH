from torch import nn
import torch.nn.functional as F
import torch
from torch.autograd import Function


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class SurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_output * tmp

        return grad_input, None


class ZO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, delta):
        out = (input > 0).float()
        L = torch.tensor([delta])
        ctx.save_for_backward(input, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, others) = ctx.saved_tensors
        delta = others[0].item()
        grad_input = grad_output.clone()

        sample_size = 5
        abs_z = torch.abs(torch.randn((sample_size,) + input.size(), device=torch.device('cuda'), dtype=torch.float))
        t = torch.abs(input[None, :, :]) < abs_z * delta
        grad_input = grad_input * torch.mean(t * abs_z, dim=0) / (2 * delta)

        return grad_input, None


class OnlineNeuron(nn.Module):
    def __init__(self, use_TEBN=False, use_SEW=False, T=4., time_slice=2.):
        super(OnlineNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.t = 0
        self.v_d = None
        self.v_s = None

        self.alpha_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)  # True
        self.beta_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.beta_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        '''
        self.alpha_1 = nn.Parameter(torch.ones(T) * torch.tensor([alpha_1]), requires_grad=True) #True 
        self.beta_1 = nn.Parameter(torch.ones(T) * torch.tensor([beta_1]), requires_grad=True) 
        self.alpha_2 = nn.Parameter(torch.ones(T) * torch.tensor([alpha_2]), requires_grad=True) 
        self.beta_2 = nn.Parameter(torch.ones(T) * torch.tensor([beta_2]), requires_grad=True) 
        '''
        self.act = SurrogateFunction.apply
        # self.act = ZO.apply
        self.gama = 1.
        self.use_TEBN = use_TEBN
        self.use_SEW = use_SEW
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self.time_slice = time_slice

    def forward(self, x):
        # print(x.shape)
        if self.use_TEBN or self.use_SEW:
            if self.use_TEBN:
                x = self.expand(x)
            if self.t == 0:
                self.v_d = torch.ones_like(x[0]) * 0. * self.v_threshold
                self.v_s = torch.ones_like(x[0]) * 0.5 * self.v_threshold
            else:
                self.v_d = self.v_d.detach()
                self.v_s = self.v_s.detach()

            spike_pot = []
            for t in range(self.T):
                self.v_d = (self.alpha_1.sigmoid() - 0.5) * self.v_d + (self.beta_1.sigmoid() - 0.5) * self.v_s + x[t]
                self.v_s = (self.alpha_2.sigmoid() + 0.5) * self.v_s + (self.beta_2.sigmoid() + 0.5) * self.v_d

                output = self.act(self.v_s - self.v_threshold, self.gama) * self.v_threshold
                self.v_s -= output.detach()
                spike_pot.append(output)

            x = torch.stack(spike_pot, dim=0)
            if self.use_TEBN:
                x = self.merge(x)
            # print(x.shape)
            return x

        else:
            if self.t == 0:
                self.v_d = torch.ones_like(x) * 0. * self.v_threshold
                self.v_s = torch.ones_like(x) * 0.5 * self.v_threshold
            if self.t % self.time_slice == 0:
                self.v_d = self.v_d.detach()
                self.v_s = self.v_s.detach()

            self.t += 1
            self.v_d = (self.alpha_1.sigmoid() - 0.5) * self.v_d + (self.beta_1.sigmoid() - 0.5) * self.v_s + x
            self.v_s = (self.alpha_2.sigmoid() + 0.5) * self.v_s + (self.beta_2.sigmoid() + 0.5) * self.v_d

            output = self.act(self.v_s - self.v_threshold, self.gama) * self.v_threshold
            self.v_s -= output.detach()
            return output

    def reset(self):
        self.t = 0


class LearnableNeuron(nn.Module):
    def __init__(self, scale=1., use_TEBN=False, use_SEW=False, T=4.):
        super(LearnableNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([scale]), requires_grad=False)
        # self.v_threshold = nn.Parameter(torch.tensor([scale]), requires_grad=True)
        self.t = 0
        self.v_d = None
        self.v_s = None

        self.alpha_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)  # True
        self.beta_1 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.beta_2 = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        '''
        self.alpha_1 = nn.Parameter(torch.ones(T) * torch.tensor([alpha_1]), requires_grad=True) #True 
        self.beta_1 = nn.Parameter(torch.ones(T) * torch.tensor([beta_1]), requires_grad=True) 
        self.alpha_2 = nn.Parameter(torch.ones(T) * torch.tensor([alpha_2]), requires_grad=True) 
        self.beta_2 = nn.Parameter(torch.ones(T) * torch.tensor([beta_2]), requires_grad=True) 
        '''
        self.act = SurrogateFunction.apply
        # self.act = ZO.apply
        self.gama = 1.
        self.use_TEBN = use_TEBN
        self.use_SEW = use_SEW
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T

    def forward(self, x):
        # print(x.shape)
        if self.use_TEBN or self.use_SEW:
            if self.use_TEBN:
                x = self.expand(x)
            self.v_d = torch.ones_like(x[0]) * 0. * (self.v_threshold)
            self.v_s = torch.ones_like(x[0]) * 0.5 * (self.v_threshold)
            spike_pot = []
            for t in range(self.T):
                self.v_d = (self.alpha_1.sigmoid() - 0.5) * self.v_d + (self.beta_1.sigmoid() - 0.5) * self.v_s + x[t]
                self.v_s = (self.alpha_2.sigmoid() + 0.5) * self.v_s + (self.beta_2.sigmoid() + 0.5) * self.v_d

                output = self.act(self.v_s - (self.v_threshold), self.gama) * (self.v_threshold)
                self.v_s -= output.detach()
                spike_pot.append(output)

            x = torch.stack(spike_pot, dim=0)
            if self.use_TEBN:
                x = self.merge(x)
            # print(x.shape)
            return x

        else:
            if self.t == 0:
                self.v_d = torch.ones_like(x) * 0. * (self.v_threshold)
                self.v_s = torch.ones_like(x) * 0.5 * (self.v_threshold)

            self.t += 1

            self.v_d = (self.alpha_1.sigmoid() - 0.5) * self.v_d + (self.beta_1.sigmoid() - 0.5) * self.v_s + x
            self.v_s = (self.alpha_2.sigmoid() + 0.5) * self.v_s + (self.beta_2.sigmoid() + 0.5) * self.v_d

            output = self.act(self.v_s -  (self.v_threshold), self.gama) *  (self.v_threshold)
            self.v_s -= output.detach()
            return output

    def reset(self):
        self.t = 0


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


qcfs = FloorLayer.apply


class QCFS(nn.Module):
    def __init__(self, up=1., t=4):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = qcfs(x * self.t + 0.5) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x


class TEBN(nn.Module):
    def __init__(self, num_features, T, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(4, 1, 1, 1, 1))
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)

    def forward(self, input):
        input = self.expand(input).transpose(0, 1).contiguous()  # T N C H W , N T C H W
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        # y = y * self.p
        y = self.merge(y)  # TNCHW  TN*CHW
        return y


class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()


class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0] / self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)


class PseudoRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x, None


pseudoRelu = PseudoRelu.apply


def softThresholdmod(x, s):
    return torch.sign(x) * pseudoRelu(torch.abs(x) - s)


def softThreshold(x, s):
    return torch.sign(x) * torch.relu(torch.abs(x) - s)


class AddDimLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1).unsqueeze(-1)


class SubDimLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.squeeze(-1).squeeze(-1)


class PConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            # self.mapping = lambda x: softThreshold(x, 0.)
            self.mapping = lambda x: softThresholdmod(x, 0.)

    def forward(self, x):
        sparseWeight = self.mapping(self.weight)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    @torch.no_grad()
    def getSparsity(self):
        sparseWeight = self.mapping(self.weight)
        temp = sparseWeight.detach().cpu()
        return (temp == 0).sum(), temp.numel()

    @torch.no_grad()
    def getSparseWeight(self):
        return self.mapping(self.weight)

    @torch.no_grad()
    def setFlatWidth(self, width):
        # self.mapping = lambda x: softThreshold(x, width)
        self.mapping = lambda x: softThresholdmod(x, width)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return PConv(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, linear=False):
    """1x1 convolution"""
    if linear is True:
        return nn.Sequential(
            AddDimLayer(),
            PConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            SubDimLayer()
        )
    else:
        return PConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
