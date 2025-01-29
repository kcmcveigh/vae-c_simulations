import torch.nn as nn
import copy
import torch


class VariationalEncoderSecondHead_modular_act_dropout(nn.Module):
    def __init__(self,
                 in_dim,
                 latent_dim,
                 encoder_hidden_dims,
                 classifier_hidden_dims,
                 encoder_activation_func,
                 classifier_activation_func,
                 n_classes,
                 dropout_rate = .2
                ):
        encoder_hidden_dims = copy.deepcopy(encoder_hidden_dims)
        # define encoder layers
        super(VariationalEncoderSecondHead_modular_act_dropout, self).__init__()
        encoder_last_layer = encoder_hidden_dims[-1] if len(encoder_hidden_dims) else in_dim
        #print(encoder_last_layer)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder =self.create_architecture_from_list(
            in_dim,
            encoder_hidden_dims,
            encoder_activation_func
        )
        self.ae_fc_mu = nn.Linear(encoder_last_layer, latent_dim)
        self.ae_fc_sigma = nn.Linear(encoder_last_layer, latent_dim)

        encoder_hidden_dims.reverse()
        final_in = encoder_hidden_dims[-1] if len(encoder_hidden_dims) else latent_dim
        #print(final_in)
        self.decoder = self.create_architecture_from_list(
            latent_dim,
            encoder_hidden_dims,
            encoder_activation_func
        )
        self.final_layer = nn.Linear(final_in, in_dim)


        self.classifier = self.create_architecture_from_list(
            latent_dim,
            classifier_hidden_dims,
            classifier_activation_func
        )
        classifier_last_layer = classifier_hidden_dims[-1] if len(classifier_hidden_dims) else latent_dim
        self.classifier_final = nn.Linear(classifier_last_layer, n_classes)
        # kl
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def create_architecture_from_list(self,
                                      input_dim,
                                      hidden_dims,
                                      activation
                                     ):
        modules=[]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    activation(),
                    self.dropout
                )
            )
            input_dim = h_dim

        return nn.Sequential(*modules)


    def encode(self, x):
        x = self.encoder(x)
        mu = self.ae_fc_mu(x)
        sigma = torch.exp(self.ae_fc_sigma(x))
        return mu, sigma

    def decode(self, x):
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    def classify(self, x):
        x = self.classifier(x)
        x = self.classifier_final(x)
        return x

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = mu + sigma * self.N.sample(mu.shape)
        x_hat = self.decode(z)
        y_pred = self.classify(z)
        self.kl = torch.mean(-0.5 * torch.sum(1 + 2 * torch.log(sigma) - mu ** 2 - sigma ** 2, dim=1), dim=0)
        return x_hat, z, y_pred
