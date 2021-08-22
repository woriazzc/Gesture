import torch
from tqdm import tqdm


def simclr_loss(out_left, out_right, tau, device='cuda'):
    N = out_left.shape[0]

    out = torch.cat((out_left, out_right), dim=0)
    norm = torch.linalg.norm(out, dim=1, keepdim=True)

    sim_matrix = torch.mm(out, out.T) / torch.mm(norm, norm.T)
    sim_exp = torch.exp(sim_matrix / tau).to(device)

    mask = (torch.ones_like(sim_exp, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    sim_exp = sim_exp.masked_select(mask).view(2*N, -1)
    exp_sum = torch.sum(sim_exp, dim=1, keepdim=True)

    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / torch.linalg.norm(out_left, dim=1, keepdim=True) / torch.linalg.norm(out_right, dim=1, keepdim=True)
    pos_pairs = torch.exp(pos_pairs / tau).repeat(2, 1).to(device)

    loss = torch.mean(-torch.log(pos_pairs / exp_sum))
    return loss


def train(model, optimizer, dataloader, epoch, epochs, temperature=0.5, batch_size=32, device='cuda'):
    model.train()
    bar = tqdm(dataloader)
    tot_loss, tot_num = 0.0, 0
    for data_pair in bar:
        x_i, x_j, lab = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)

        _, z_i = model(x_i)
        _, z_j = model(x_j)
        loss = simclr_loss(z_i, z_j, temperature, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item() * batch_size
        tot_num += batch_size
        bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, tot_loss / tot_num))

    return tot_loss / tot_num


def test(model, memory_dataloader, test_dataloader, k, epoch, epochs, C, temperature=0.5, device='cuda'):
    model.eval()
    feature_bank = []
    feature_label = []
    with torch.no_grad():
        for x, _, target in tqdm(memory_dataloader, desc='Feature extracting'):
            feature, out = model(x.to(device))
            feature_bank.append(feature)
            feature_label.append(target)

        # (D, N)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # (N)
        feature_label = torch.cat(feature_label).to(device)

        N = feature_bank.shape[1]

        total_top1, total_top2, total_num = 0.0, 0.0, 0
        test_bar = tqdm(test_dataloader)
        for x, _, target in test_bar:
            # (B, *), (B)
            x, target = x.to(device), target.to(device)

            # (B, D)
            feature, out = model(x)

            # (B, N)
            sim_matrix = torch.mm(feature, feature_bank)

            # (B, K)   (B, K)
            fk_weight, fk_index = sim_matrix.topk(k=k, dim=-1)
            fk_weight = (fk_weight / temperature).exp()

            # (B, K)
            sim_labels = torch.gather(feature_label.expand(N, -1), 1, fk_index)
            B, K = sim_labels.shape

            # (B, K, C)
            one_hot = torch.zeros(B, K, C, device=device)
            one_hot = one_hot.scatter(-1, sim_labels.unsqueeze(-1), 1)

            # (B, K)
            pred_scores = torch.sum(one_hot * fk_weight.unsqueeze(-1), dim=1)

            # (B, K)
            pred_labels = pred_scores.argsort(dim=1, descending=True)

            total_num += B
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(-1)).any(dim=1).float()).item()
            total_top2 += torch.sum((pred_labels[:, :2] == target.unsqueeze(-1)).any(dim=1).float()).item()

            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@2:{:.2f}%'.format(epoch, epochs, total_top1 / total_num * 100, total_top2 / total_num * 100))

    return total_top1 / total_num * 100, total_top2 / total_num * 100, feature_bank, feature_label
