import argparse

import onnx
import torch
import torch.nn as nn
import torchvision.models as models
from onnxoptimizer import optimize
from onnxsim import simplify
from torchvision.models import MobileNet_V3_Large_Weights


class LandmarkModel(nn.Module):
    def __init__(self, base_model, num_features, num_landmarks):
        super().__init__()
        self.base = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # Все кроме classifier
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_landmarks * 2),
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x


def export(load_checkpoint: str, output_name: str) -> None:
    base_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    num_ftrs = base_model.classifier[0].in_features

    model = LandmarkModel(base_model, num_ftrs, 68)

    model.load_state_dict(torch.load(load_checkpoint))

    model.eval()

    example_inputs = torch.randn(1, 3, 256, 256)
    # Экспорт в ONNX
    torch.onnx.export(
        model,  # модель
        example_inputs,  # пример входа
        output_name,  # путь к файлу
        export_params=True,  # сохранить веса в модели
        input_names=["input"],  # имена входов
        output_names=["output"],  # имена выходов
        opset_version=20,  # for affine transform
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def optimize_onnx(model_path: str, optimized_model_path: str):
    model = onnx.load(model_path)

    # Оптимизация
    optimized_model = optimize(model)

    # Упрощение
    optimized_model, _ = simplify(optimized_model)

    onnx.save(model, optimized_model_path)


def validate(model_path: str):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("exported model is ok")
    except Exception as e:
        print("exported model error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="onnx export script")

    parser.add_argument("-model", type=str, help="path to torch model")
    parser.add_argument("-onnx", type=str, help="path to onnx model")

    args = parser.parse()

    torch_model = args.model
    onnx_model = args.onnx

    export(torch_model, onnx_model)
    optimize_onnx(onnx_model, onnx_model)
    validate(onnx_model)
