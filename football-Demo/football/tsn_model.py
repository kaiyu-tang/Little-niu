from torch import nn
import torchvision

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model, dropout=0.8):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.num_class = num_class
        self.reshape = True
        self.dropout = dropout
        self.softmax = nn.Softmax()
        self._prepare_tsn_model(base_model)


    def _prepare_tsn_model(self, base_model):
        self.base_model = getattr(torchvision.models, base_model)()
        self.base_model.last_layer_name = 'fc'
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, self.num_class)


    def forward(self, input):
        sample_len = 3
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = base_out.mean(dim=1, keepdim=True)
        return output.squeeze(1)