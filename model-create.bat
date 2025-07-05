REM ===== Pull base model ====
ollama pull qwen2.5:1.5b
ollama pull llama3.2:latest
ollama pull gemma3:1b


REM ===== Remove Model =====
ollama rm pdf-qwen 
ollama rm pdf-llama
ollama rm pdf-gemma



REM ===== Create Model =====
ollama create pdf-qwen -f Modelfile-qwen
ollama create pdf-llama -f Modelfile-llama
ollama create pdf-gemma -f Modelfile-gemma
