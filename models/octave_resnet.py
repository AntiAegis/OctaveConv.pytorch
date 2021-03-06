#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from base import BaseBackbone


#------------------------------------------------------------------------------
#  OctConv
#------------------------------------------------------------------------------
class OctConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
					alpha_in=0.25, alpha_out=0.25, type='normal'):
		
		# Initialize
		super(OctConv, self).__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.type = type

		hf_ch_in = int(in_channels * (1 - alpha_in))
		hf_ch_out = int(out_channels * (1 - alpha_out))
		lf_ch_in = in_channels - hf_ch_in
		lf_ch_out = out_channels - hf_ch_out

		# Downsampling
		if stride == 2:
			self.downsample = nn.AvgPool2d(kernel_size=2, stride=stride)

		if type == 'first':
			self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
			self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size, padding=padding)
			self.convl = nn.Conv2d(in_channels, lf_ch_out, kernel_size=kernel_size, padding=padding)
		elif type == 'last':
			self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
			self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding)
			self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding)
		elif type == 'normal':
			self.avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)
			self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
			self.L2L = nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=kernel_size, padding=padding)
			self.L2H = nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=kernel_size, padding=padding)
			self.H2L = nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=kernel_size, padding=padding)
			self.H2H = nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=kernel_size, padding=padding)
		else:
			raise NotImplementedError


	def forward(self, x):
		if self.type == 'first':
			x = self.downsample(x) if self.stride==2 else x
			hf_out = self.convh(x)
			lf_out = self.convl(self.avg_pool(x))
			return hf_out, lf_out

		elif self.type == 'last':
			hf, lf = x
			if self.stride == 2:
				hf_out = self.convh(self.downsample(hf))
				lf_out = self.convl(lf)
				return hf_out + lf_out
			else:
				hf_out = self.convh(hf)
				lf_out = self.convl(self.upsample(lf))
				return hf_out + lf_out
		else:
			hf, lf = x
			if self.stride == 2:
				hf = self.downsample(hf)
				hf_out = self.H2H(hf) + self.L2H(lf)
				lf_out = self.L2L(self.avg_pool(lf)) + self.H2L(self.avg_pool(hf))
				return hf_out, lf_out
			else:
				hf_out = self.H2H(hf) + self.upsample(self.L2H(lf))
				lf_out = self.L2L(lf) + self.H2L(self.avg_pool(hf))
				return hf_out, lf_out


#------------------------------------------------------------------------------
#  Fundamental convolutions
#------------------------------------------------------------------------------
def norm_conv3x3(in_planes, out_planes, stride=1, type=None):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def norm_conv1x1(in_planes, out_planes, stride=1, type=None):
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def oct_conv3x3(in_planes, out_planes, stride=1, type='normal'):
	return OctConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, type=type)

def oct_conv1x1(in_planes, out_planes, stride=1, type='normal'):
	return OctConv(in_planes, out_planes, kernel_size=1, stride=stride, type=type)


#------------------------------------------------------------------------------
#  OctBatchNorm and OctReLu
#------------------------------------------------------------------------------
class OctBatchNorm(nn.Module):
	def __init__(self, num_features, alpha_in=0.25, alpha_out=0.25,
				eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
		# Initialize
		super(OctBatchNorm, self).__init__()
		hf_ch = int(num_features * (1 - alpha_in))
		lf_ch = num_features - hf_ch
		self.bnh = nn.BatchNorm2d(hf_ch, eps, momentum, affine, track_running_stats)
		self.bnl = nn.BatchNorm2d(lf_ch, eps, momentum, affine, track_running_stats)

	def forward(self, x):
		hf, lf = x
		return self.bnh(hf), self.bnl(lf)


class OctReLu(nn.ReLU):
	def forward(self, x):
		hf, lf = x
		hf = super(OctReLu, self).forward(hf)
		lf = super(OctReLu, self).forward(lf)
		return hf, lf


#------------------------------------------------------------------------------
#  BasicBlock
#------------------------------------------------------------------------------
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, type="normal", oct_conv_on=True):
		super(BasicBlock, self).__init__()
		conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
		norm_func = OctBatchNorm if oct_conv_on else nn.BatchNorm2d
		act_func = OctReLu if oct_conv_on else nn.ReLU

		self.conv1 = conv3x3(inplanes, planes, type="first" if type == "first" else "normal")
		self.bn1 = norm_func(planes)
		self.relu1 = act_func(inplace=True)
		self.conv2 = conv3x3(planes, planes, stride, type="last" if type == "last" else "normal")
		if type == "last":
			norm_func = nn.BatchNorm2d
			act_func = nn.ReLU
		self.bn2 = norm_func(planes)
		self.relu2 = act_func(inplace=True)
		self.downsample = downsample
		self.stride = stride


	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		if isinstance(out, (tuple, list)):
			assert len(out) == len(identity) and len(out) == 2
			out = (out[0] + identity[0], out[1] + identity[1])
		else:
			out += identity

		out = self.relu2(out)
		return out


#------------------------------------------------------------------------------
#   Bottleneck
#------------------------------------------------------------------------------
class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, type="normal", oct_conv_on=True):
		super(Bottleneck, self).__init__()
		conv1x1 = oct_conv1x1 if oct_conv_on else norm_conv1x1
		conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
		norm_func = OctBatchNorm if oct_conv_on else nn.BatchNorm2d
		act_func = OctReLu if oct_conv_on else nn.ReLU

		self.conv1 = conv1x1(inplanes, planes, type="first" if type == "first" else "normal")
		self.bn1 = norm_func(planes)
		self.relu1 = act_func(inplace=True)
		self.conv2 = conv3x3(planes, planes, stride, type="last" if type == "last" else "normal")
		if type == "last":
			conv1x1 = norm_conv1x1
			norm_func = nn.BatchNorm2d
			act_func = nn.ReLU
		self.bn2 = norm_func(planes)
		self.relu2 = act_func(inplace=True)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = norm_func(planes * self.expansion)
		self.relu3 = act_func(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		if isinstance(out, (tuple, list)):
			assert len(out) == len(identity) and len(out) == 2
			out = (out[0] + identity[0], out[1] + identity[1])
		else:
			out += identity

		out = self.relu3(out)
		return out


#------------------------------------------------------------------------------
#   ResNet
#------------------------------------------------------------------------------
class ResNet(BaseBackbone):
	def __init__(self, block, layers, num_classes=1000, use_octave=False, zero_init_residual=False):
		super(ResNet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], type="first")
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, type="last")
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		self.init_weights()
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)


	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


	def _make_layer(self, block, planes, blocks, stride=1, type="normal"):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion or type=='first':
			norm_func = nn.BatchNorm2d if type == "last" else OctBatchNorm
			downsample = nn.Sequential(
				oct_conv1x1(self.inplanes, planes * block.expansion, stride, type=type),
				norm_func(planes * block.expansion))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, type=type))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, oct_conv_on=type != "last"))
		return nn.Sequential(*layers)


#------------------------------------------------------------------------------
#   Original instances
#------------------------------------------------------------------------------
def resnet18(pretrained=False, **kwargs):
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	return model


#------------------------------------------------------------------------------
#   Octave instances
#------------------------------------------------------------------------------
def octave_resnet18(pretrained=False, **kwargs):
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	return model

def octave_resnet34(pretrained=False, **kwargs):
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	return model

def octave_resnet50(pretrained=False, **kwargs):
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	return model

def octave_resnet101(pretrained=False, **kwargs):
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	return model

def octave_resnet152(pretrained=False, **kwargs):
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	return model