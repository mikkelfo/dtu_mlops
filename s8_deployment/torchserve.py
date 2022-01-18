import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')


input = torch.randn(64,3,7,7)

unscripted = model(input)
scripted = script_model(input)


assert torch.allclose(unscripted, scripted)