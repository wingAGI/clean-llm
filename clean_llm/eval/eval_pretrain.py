import torch

def evaluate(
    model,
    tokenizer,
    device,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    eos_token_id,
):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output_tokens = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
    output_ids = output_tokens[0].cpu().numpy().tolist()
    full_ids = input_ids + output_ids
    text = tokenizer.decode(full_ids)
    return text


