import torch
import torch.nn as nn


class HSI_ViT(nn.Module):

    def __init__(
        self,
        input_channels=30,
        patch_size=11,
        num_classes=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size

        # --------------------------------
        # 1. Convert spectral features
        #    30 -> 64
        # --------------------------------

        self.embedding = nn.Linear(
            input_channels,
            embed_dim
        )

        # --------------------------------
        # 2. CLS token
        # --------------------------------

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        # --------------------------------
        # 3. Positional embeddings
        # --------------------------------

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_patches + 1,
                embed_dim
            )
        )

        # --------------------------------
        # 4. Transformer Encoder
        # --------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # --------------------------------
        # 5. Classification head
        # --------------------------------

        self.classifier = nn.Linear(
            embed_dim,
            num_classes
        )

    def forward(self, x):

        # Input:
        # [batch, 11, 11, 30]

        batch_size = x.size(0)

        # --------------------------------
        # Convert 11×11 spatial locations
        # into 121 tokens
        # --------------------------------

        x = x.reshape(
            batch_size,
            self.num_patches,
            -1
        )

        # Shape:
        # [batch, 121, 30]

        # --------------------------------
        # Spectral feature embedding
        # --------------------------------

        x = self.embedding(x)

        # Shape:
        # [batch, 121, 64]

        # --------------------------------
        # Add CLS token
        # --------------------------------

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1
        )

        # Shape:
        # [batch, 122, 64]

        # --------------------------------
        # Add positional information
        # --------------------------------

        x = x + self.pos_embedding

        # --------------------------------
        # Transformer
        # --------------------------------

        x = self.transformer(x)

        # --------------------------------
        # Take CLS token
        # --------------------------------

        cls_output = x[:, 0]

        # Shape:
        # [batch, 64]

        # --------------------------------
        # Classification
        # --------------------------------

        output = self.classifier(cls_output)

        # Shape:
        # [batch, 16]

        return output


# --------------------------------
# Test the model
# --------------------------------

if __name__ == "__main__":

    model = HSI_ViT()

    print(model)

    # Create dummy HSI batch
    x = torch.randn(
        4,
        11,
        11,
        30
    )

    output = model(x)

    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)

# End of module.
