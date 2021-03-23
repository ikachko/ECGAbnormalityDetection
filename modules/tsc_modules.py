from modules.base import BaseModel
from models import inception


class InceptionTime(BaseModel):

    def __init__(self, hparams, model_params):
        super().__init__(hparams, model_params)

        self.model = inception.InceptionModel(
                        num_blocks=model_params['num_blocks'],
                        in_channels=model_params['in_channels'],
                        out_channels=model_params['out_channels'],
                        bottleneck_channels=model_params['bottleneck_channels'],
                        kernel_sizes=model_params['kernel_sizes'],
                        use_residuals=model_params['use_residuals'],
                        num_pred_classes=model_params['num_pred_classes']
                        )