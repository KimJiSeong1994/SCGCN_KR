def model_train(model, data_loader, loss_fn, optimizer, device):
    from tqdm import tqdm
    model.train()
    running_loss, corr= 0, 0
    prograss_bar = tqdm(data_loader)

    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()
        _, pred = output.max(dim = 1)
        corr += pred.eq(lbl).sum().item()
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):
    import torch
    model.eval()
    with torch.no_grad():
        corr, running_loss = 0, 0

        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc
