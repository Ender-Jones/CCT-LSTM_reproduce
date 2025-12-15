import torch
import torch.nn as nn
from vit_pytorch.cct import CCT


class CCTForPreTraining(nn.Module):
    """
    Wrapper for the CCT model for the pre-training phase (Stage 1).
    This model takes a single MTF image and performs classification.
    """

    def __init__(self,
                 img_size=(224, 224),
                 embedding_dim=256,
                 n_conv_layers=2,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 num_layers=6,
                 num_heads=4,
                 mlp_ratio=2.,
                 num_classes=3,
                 positional_embedding='learnable',
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 *args, **kwargs):
        """
        Initializes the CCT model for pre-training.

        Args:
            img_size (tuple): The size of the input image (height, width).
            embedding_dim (int): The dimension of the token embeddings.
            n_conv_layers (int): The number of convolutional layers in the tokenizer.
            kernel_size (int): The kernel size for the convolutional tokenizer.
            stride (int): The stride for the convolutional tokenizer.
            padding (int): The padding for the convolutional tokenizer.
            pooling_kernel_size (int): The kernel size for the pooling layer.
            pooling_stride (int): The stride for the pooling layer.
            pooling_padding (int): The padding for the pooling layer.
            num_layers (int): The number of transformer blocks.
            num_heads (int): The number of attention heads.
            mlp_ratio (float): The ratio of the MLP hidden dimension to the embedding dimension.
            num_classes (int): The number of output classes for the final classifier.
            positional_embedding (str): The type of positional embedding to use.
        """
        super().__init__()
        self.cct = CCT(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            dropout=dropout,
            emb_dropout=emb_dropout,
            *args, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the pre-training CCT.
        Args:
            x (torch.Tensor): A batch of images with shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        return self.cct(x)


class CCT_LSTM_Model(nn.Module):
    """
    The final CCT-LSTM model for the end-to-end training phase (Stage 2).
    This model processes sequences of MTF image pairs (landmark and rPPG).
    """

    def __init__(self,
                 cct_params: dict,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 2,
                 num_classes: int = 3):
        """
        Initializes the full CCT-LSTM model.

        Args:
            cct_params (dict): A dictionary of parameters to initialize the CCT backbones.
                               Should not include 'num_classes'.
            lstm_hidden_size (int): The number of features in the LSTM hidden state.
            lstm_num_layers (int): The number of recurrent layers in the LSTM.
            num_classes (int): The number of output classes for the final classifier.
        """
        super().__init__()
        # 1. Initialize two CCT backbones, one for each modality.
        # Configure CCT to output embeddings directly with dimension = embedding_dim
        cct_embedding_dim = cct_params.get('embedding_dim', 256)
        self.cct_landmark = CCT(num_classes=cct_embedding_dim, **cct_params)
        self.cct_rppg = CCT(num_classes=cct_embedding_dim, **cct_params)
        # Replace classification head with Identity so the network outputs pooled embeddings
        self.cct_landmark.mlp_head = nn.Identity()
        self.cct_rppg.mlp_head = nn.Identity()
        # The input to the LSTM will be the concatenated features from both CCTs.
        lstm_input_size = cct_embedding_dim * 2

        # 2. Initialize the LSTM layer.
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True  # This is crucial for handling (batch, seq, feature) tensors
        )

        # 3. Initialize the final classifier.
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def load_pretrained_weights(self, landmark_weights_path: str, rppg_weights_path: str):
        """
        Loads pre-trained weights from Stage 1 into the CCT backbones.
        """
        # Load landmark CCT weights
        print(f"Loading pre-trained landmark weights from: {landmark_weights_path}")
        try:
            landmark_state_dict = torch.load(landmark_weights_path, weights_only=True)
        except TypeError:
            # fallback for older torch versions without weights_only
            landmark_state_dict = torch.load(landmark_weights_path)
        # Filter out the mlp_head weights from the pre-trained model
        landmark_filtered_dict = {k: v for k, v in landmark_state_dict.items() if not k.startswith('cct.mlp_head')}
        self.cct_landmark.load_state_dict(landmark_filtered_dict, strict=False)

        # Load rPPG CCT weights
        print(f"Loading pre-trained rPPG weights from: {rppg_weights_path}")
        try:
            rppg_state_dict = torch.load(rppg_weights_path, weights_only=True)
        except TypeError:
            rppg_state_dict = torch.load(rppg_weights_path)
        # Filter out the mlp_head weights
        rppg_filtered_dict = {k: v for k, v in rppg_state_dict.items() if not k.startswith('cct.mlp_head')}
        self.cct_rppg.load_state_dict(rppg_filtered_dict, strict=False)
        print("Pre-trained weights loaded successfully.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CCT-LSTM model.

        Args:
            x (torch.Tensor): A batch of video sequences with shape
                              (batch_size, sequence_length, 2, channels, height, width).
                              The dimension with size 2 corresponds to the (landmark, rppg) modalities.

        Returns:
            torch.Tensor: The final output logits from the classifier for the entire sequence.
        """
        batch_size, seq_len, _, c, h, w = x.shape

        # 1. Separate the two modalities
        # Landmark: (batch_size, seq_len, C, H, W)
        # rPPG: (batch_size, seq_len, C, H, W)
        landmark_seq = x[:, :, 0, :, :, :]
        rppg_seq = x[:, :, 1, :, :, :]

        # 2. Reshape for CCT processing.
        # CCT expects a batch of images (B, C, H, W). We treat the sequence length
        # as part of the batch dimension for feature extraction.
        # New shape: (batch_size * seq_len, C, H, W)
        landmark_flat = landmark_seq.reshape(batch_size * seq_len, c, h, w)
        rppg_flat = rppg_seq.reshape(batch_size * seq_len, c, h, w)

        # 3. Extract features using the CCT backbones.
        # Output shape: (batch_size * seq_len, embedding_dim)
        landmark_features = self.cct_landmark(landmark_flat)
        rppg_features = self.cct_rppg(rppg_flat)

        # 4. Concatenate the features from both modalities.
        # Output shape: (batch_size * seq_len, embedding_dim * 2)
        combined_features = torch.cat([landmark_features, rppg_features], dim=1)

        # 5. Reshape the features back into a sequence for the LSTM.
        # New shape: (batch_size, seq_len, embedding_dim * 2)
        feature_sequence = combined_features.view(batch_size, seq_len, -1)

        # 6. Process the sequence through the LSTM.
        # We only need the final hidden state for classification.
        # h_n shape: (num_layers, batch_size, hidden_size)
        _, (h_n, _) = self.lstm(feature_sequence)

        # 7. Use the hidden state of the last layer for classification.
        # We take the output from the last LSTM layer: (batch_size, hidden_size)
        last_layer_hidden_state = h_n[-1]

        # 8. Pass through the final classifier.
        # Output shape: (batch_size, num_classes)
        output = self.classifier(last_layer_hidden_state)

        return output
