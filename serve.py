from numpy import dtype
from axon_serve import PredictionService, GRPCService
import torch
import yaml
import os

from autoencoder.autoencoder import Autoencoder

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

autoencoder_config = config["autoencoder"]


class TestPredictionService(PredictionService):
    def __init__(self):
        super().__init__()
        self.autoencoder = Autoencoder(autoencoder_config)

    def predict(self, model_input, params):

        print("model_input: ", model_input.shape)

        model_input = model_input.copy()
        model_input = torch.from_numpy(model_input).float()
        model_out = self.autoencoder(model_input, inference=True)

        print("model_out: ", model_out.shape)
        print("params: ", params)

        return model_out


if __name__ == "__main__":
    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    test_prediction_service = TestPredictionService()
    service = GRPCService(test_prediction_service, port=8000)

    service.start()
