import torch

def autoregressive_predict(model, initial_seq, pred_len=60, device='cuda'):
    model.eval()
    generated = []

    input_seq = initial_seq.unsqueeze(0).to(device)  # [1, seq_len, 3]

    with torch.no_grad():
        for _ in range(pred_len):
            output = model(input_seq)  # [1, seq_len, 3]
            next_delta = output[:, -1, :]  # [1, 3]
            generated.append(next_delta.cpu())

            last_input = input_seq[0, 1:, :].clone()  # [seq_len-1, 3]
            new_row = next_delta  # [1, 3]
            input_seq = torch.cat([last_input, new_row], dim=0).unsqueeze(0)  # [1, seq_len, 3]

    return torch.stack(generated).squeeze(1)  # [pred_len, 3]