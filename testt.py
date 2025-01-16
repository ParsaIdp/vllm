from vllm import LLM

# Initialize the model
llm = LLM(
    model="LlamaPRMSearchForCausalLM",
    model_path="path/to/llama-checkpoint",
    gpu_memory_utilization=0.8,
)

# Generate text with beam search
output = llm.generate_with_beam_search("What are the applications of AI?", num_beams=3, max_length=50)
print("Beam Search Output:", output)
