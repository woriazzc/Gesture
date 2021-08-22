import torch

class Gesture:
    def __init__(self, model_path, feature_bank_path, feature_label_path, k=200, C=5, temperature=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.C = C
        self.temperature = temperature
        self.model = torch.load(model_path).to(self.device)
        self.feature_bank = torch.load(feature_bank_path).to(self.device)
        self.feature_label = torch.load(feature_label_path).to(self.device)
        self.model.eval()

    def predict(self, img):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        img = img.unsqueeze(0)
        with torch.no_grad():
            img = img.to(self.device)

            # (1, D)
            feature, out = self.model(img)

            # (1, N)
            sim_matrix = torch.mm(feature, self.feature_bank)

            # (1, K)   (1, K)
            fk_weight, fk_index = sim_matrix.topk(k=self.k, dim=-1)
            fk_weight = (fk_weight / self.temperature).exp()

            # (1, K)
            sim_labels = torch.gather(self.feature_label.expand(1, -1), 1, fk_index)
            B, K = sim_labels.shape

            # (1, K, C)
            one_hot = torch.zeros(B, K, self.C, device=self.device)
            one_hot = one_hot.scatter(-1, sim_labels.unsqueeze(-1), 1)

            # (1, K)
            pred_scores = torch.sum(one_hot * fk_weight.unsqueeze(-1), dim=1)

            # (1, K)
            pred_labels = pred_scores.argsort(dim=1, descending=True)

        return pred_labels[0, 0].item()